import sys
import os

# 添加当前目录到 Python 路径
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


class Ur5eEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml

    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]


        self.ee_body_name = "wrist_3_link"
        self.arm = pinocchio_kinematic.Kinematics("wrist_3_link")
        self.arm.buildFromMJCF(self.arm_xml)

        self.last_dof = self.data.ctrl[:6].copy()
        self.setTimestep(0.05)

        # 导纳状态变量：全部在向量空间/切空间定义(只考虑z方向)
        self.delta_d_ee_des_z = 0  # 速度增量 [v, omega]
        self.delta_ee_des_z = 0
        self.first_goto_initial_pos_cnt = 100

        #沿y轴方向移动速度
        self.v_y=0.01
        self.delta_d_ee_des_y=0
        self.delta_ee_des_y=0

        #微分运动学求解
        self.dd_ee_des=np.zeros(6)
        self.delta_d_ee_des=np.zeros(6)


        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)

        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)

        # 添加力传感器绘图器
        self.plot_manager.addNewFigurePlotter("force", title="力传感器 - 力", row=0, col=1)

        # 为力传感器数据添加带颜色的曲线

        self.plot_manager.addPlotToPlotter("force", "force.z", color=(0, 0, 255))  # 蓝色

        # 获取力传感器 ID
        mujoco.mj_forward(self.model, self.data)
        # 尝试通过名称查找传感器，如果失败则查找第一个力传感器
        self.force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor_site")
        if self.force_sensor_id < 0:
            # 如果名称查找失败，查找第一个力传感器
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
                    self.force_sensor_id = i
                    break
        self.ik_stop = False



    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            # 强行保持初始位置
            self.data.qpos[:6] = self.initial_pos[:6]
            self.data.qvel[:6] = 0
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:6] = self.data.qfrc_bias[:6].copy()
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.last_ee_pos = self.ee_pos.copy()
            self.start_ee_pos = self.ee_pos.copy()
            self.desired_pos = self.ee_pos.copy()
        else:
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            print(self.now_ee_pos)
            dt = self.model.opt.timestep
            # 获取 Site 和 Body ID
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "force_sensor_site")
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
            now_ee_vel = (self.now_ee_pos - self.last_ee_pos) / dt
            now_ee_pos_z=self.now_ee_pos[2]
            last_ee_pos_z=self.last_ee_pos[2]
            desired_pos_z=self.desired_pos[2]
            ee_pos_err = now_ee_pos_z - desired_pos_z

            # --- 2. 运动学计算 (Jacobian & Velocity) ---
            # 2.1 计算雅可比矩阵
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))

            site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "force_sensor_site"
            )

            mujoco.mj_jacSite(
                self.model,
                self.data,
                jac_pos,
                jac_rot,
                site_id
            )

            J = np.vstack((jac_pos, jac_rot))
            J_task = J[:, :6]
            current_twist = J_task @ self.data.qvel[:6]



            # 3. 读取力传感器数据（需要在导纳控制之前读取）
            sensor_addr = self.model.sensor_adr[self.force_sensor_id]
            force_z =self.data.sensordata[sensor_addr+2]

            # 5. 导纳动力学方程

            self.M_d = 1
            self.B_d = 1000
            self.K_d = 0

            F_target = 20 # 施加虚拟力（z方向上的）
            print("force_z=",force_z)
            f_error = F_target-force_z



            dd_ee_z = (1.0/self.M_d) * (f_error - self.B_d * current_twist[2] - self.K_d * ee_pos_err)

            #print(dd_ee_z)


            #6. 积分更新期望位姿  𝜏𝑑 = 𝐽^T(𝑞)𝛼 + 𝐹^𝑓̇𝑞 + ̂𝑔(𝑞)

            #6.1  𝐽^T(𝑞) 雅可比矩阵设置


            # 计算雅可比
            damping=1
            J_inv = J_task.T @ np.linalg.inv(J_task @ J_task.T + damping ** 2 * np.eye(6))

            # 6.2 𝐹^𝑓 粘性摩擦系数矩阵设置
            # 假设 UR5e 每个关节的粘性摩擦系数 (需要通过实验微调)
            # 通常大臂关节(0,1,2)系数大，手腕关节(3,4,5)系数小
            friction_coeffs = np.array([5.0, 5.0, 2.0,0.5,0.5, 0.5])
            F_f = np.diag(friction_coeffs)

            # 计算当前的摩擦扭矩
            tau_friction = F_f @ self.data.qvel[:6]


            #6.3̂ 𝑔(𝑞) 重力补偿系数矩阵设置 (包含重力、离心力、科里奥利力)
            bias = self.data.qfrc_bias[:6].copy()


            # 6.4 最终计算
            # 加速度（alpha）
            self.dd_ee_des[2] = dd_ee_z
            print("self.dd_ee_des=", self.dd_ee_des)

            #alpha
            alpha=np.zeros(6)
            alpha[2]= self.M_d * dd_ee_z + self.B_d * current_twist[2] + self.K_d * ee_pos_err

            tau_d =  J_task.T @ alpha + tau_friction + bias
            tau_d = np.clip(tau_d, -50, 50)
            print("tau_d=",tau_d)
            #print("tau_d=",tau_d)



            # 7 执行

            self.data.ctrl[:6] = tau_d
            self.last_ee_pos = self.now_ee_pos.copy()

            # 8 绘图调试
            self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
            self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])

            # 绘制力传感器数据
            self.plot_manager.updateDataToPlotter("force", "force.z", force_z)





if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e_motor.xml"
    env = Ur5eEnv(SCENE_XML, ARM_XML)
    env.run_loop()