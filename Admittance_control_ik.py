import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pinocchio as pin
import src.pinocchio_kinematic as pinocchio_kinematic
import src.utils as utils
import src.lowpass_filter as lowpass_filter

import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d as tf
import warnings  # 新增：导入警告模块

# 新增：关闭ikpy的fixed link警告
warnings.filterwarnings("ignore", category=UserWarning, module="ikpy.chain")

class Ur5eEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml

        # -------------------------- 原有初始化逻辑 --------------------------
        self.initial_pos = self.model.key_qpos[0]
        self.last_pos_angle = self.initial_pos[:6].copy()  # 修改1：只保留6个运动关节角度
        self.ee_body_name = "wrist_3_link"
        self.arm = pinocchio_kinematic.Kinematics("wrist_3_link")
        self.arm.buildFromMJCF(self.arm_xml)

        self.last_dof = self.data.ctrl[:6].copy()
        self.setTimestep(0.005)

        # 导纳状态变量：全部在向量空间/切空间定义(只考虑z方向)
        self.delta_ee_des_z = 0
        self.delta_d_ee_des_z = 0
        self.first_goto_initial_pos_cnt = 100

        # 沿y轴方向移动速度
        self.delta_d_ee_des_y = 0
        self.delta_ee_des_y = 0

        # 微分运动学求解
        self.dd_ee_des = np.zeros(6)
        self.delta_d_ee_des = np.zeros(6)

        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)

        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)

        # 添加力传感器绘图器
        self.plot_manager.addNewFigurePlotter("force", title="力传感器 - 力", row=0, col=1)
        self.plot_manager.addPlotToPlotter("force", "force.z", color=(0, 0, 255))  # 蓝色

        # 获取力传感器 ID
        mujoco.mj_forward(self.model, self.data)
        self.force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor_site")
        if self.force_sensor_id < 0:
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
                    self.force_sensor_id = i
                    break
        self.ik_stop = False

        # 修改2：初始化时创建ikpy的chain，指定正确的active_links_mask（关键！）
        # UR5e的URDF链接索引：0(fixed)、1-6(运动关节)、7(fixed)
        self.active_links_mask = [
            False,  # 0: Base link (fixed)
            True,   # 1: shoulder_pan_joint
            True,   # 2: shoulder_lift_joint
            True,   # 3: elbow_joint
            True,   # 4: wrist_1_joint
            True,   # 5: wrist_2_joint
            True,   # 6: wrist_3_joint
            False   # 7: ee_fixed_joint (fixed)
        ]
        # 只创建一次chain，避免每次循环重复加载
        self.my_chain = ikpy.chain.Chain.from_urdf_file(
            "model/ur5e.urdf",
            active_links_mask=self.active_links_mask  # 指定mask，只激活6个运动关节
        )

    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            # 强行保持初始位置
            self.data.qpos[:6] = self.initial_pos[:6]
            self.data.qvel[:6] = 0
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:6] = self.data.qfrc_bias[:6].copy()
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.last_ee_pos = np.zeros(6)
            self.last_ee_pos[0:3] = self.ee_pos[0:3].copy()
            self.start_ee_pos = np.zeros(6)
            self.desired_pos = np.zeros(6)
            self.start_ee_pos[:] = self.ee_pos[:]
            self.desired_pos[:] = self.ee_pos[:]
        else:
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            dt = self.model.opt.timestep
            # 获取 Site 和 Body ID
            ee_vel_ext_z=(self.desired_pos[2]-self.last_ee_pos[2])/dt
            now_ee_vel_z = (self.now_ee_pos[2] - self.last_ee_pos[2]) / dt
            now_ee_pos_z = self.now_ee_pos[2]
            last_ee_pos_z = self.last_ee_pos[2]
            desired_pos_z = self.desired_pos[2]

            ee_pos_err_z = now_ee_pos_z - last_ee_pos_z

            # 3. 读取力传感器数据（需要在导纳控制之前读取）
            f_site = self.data.sensordata[0:3]
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "force_sensor_site")
            R_site2world = self.data.site_xmat[site_id].reshape(3, 3)
            f_world = R_site2world @ f_site
            force_z_world = f_world[2]

            # 5. 导纳动力学方程
            self.M_d = 1
            self.B_d = 1000
            self.K_d = 0

            F_target = -20  # 施加虚拟力（z方向上的）
            print("force_z=", force_z_world)
            f_error = force_z_world - F_target
            dd_ee_z = (1.0 / self.M_d) * (f_error - self.B_d * now_ee_vel_z - self.K_d * ee_pos_err_z)

            self.delta_d_ee_des_z = dd_ee_z * dt
            self.delta_ee_des_z = self.delta_d_ee_des_z * dt
            # 修改3：desired_pos[2]基于初始位置累加，避免数值异常
            self.desired_pos[2] = self.start_ee_pos[2] + self.delta_ee_des_z
            print("desired_z=", self.desired_pos[2])

            # 沿y轴运动
            v_y = -0.005
            self.delta_d_ee_des_y += v_y *dt
            self.desired_pos[1] = self.start_ee_pos[1] + self.delta_d_ee_des_y
            print("desired_pos=", self.desired_pos[:3])

            # ik求解（修改5：使用初始化时创建的chain，且初始角度长度匹配）
            ee_orientation = tf.euler.euler2mat(*self.desired_pos[3:6])  # 欧拉角转旋转矩阵
            ref_pos=np.zeros(8)
            ref_pos[1:7] = self.last_pos_angle[0:6]
            # 关键：initial_position是长度6的数组，和active_links_mask的活跃关节数匹配
            pos_angle = self.my_chain.inverse_kinematics(
                target_position=self.desired_pos[:3],
                target_orientation=ee_orientation,
                orientation_mode="all",
                initial_position=ref_pos
            )
            # 修改6：只取6个运动关节的角度（ikpy返回的pos_angle长度8，索引1-6是运动关节）
            ctrl_angles = pos_angle[1:7]  # 跳过索引0（fixed），取1-6共6个关节角度
            self.data.ctrl[:6] = ctrl_angles

            # 修改7：更新last_pos_angle为6个运动关节角度，保持长度一致
            self.last_pos_angle = ctrl_angles.copy()
            self.last_ee_pos = self.now_ee_pos.copy()  # 修改8：更新last_ee_pos为当前实际位置，而非desired_pos

            # 8 绘图调试
            self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
            self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
            self.plot_manager.updateDataToPlotter("force", "force.z", force_z_world)

if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e_2.xml"
    env = Ur5eEnv(SCENE_XML, ARM_XML)
    env.run_loop()