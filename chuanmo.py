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
        self.arm_xml = arm_xml
        self.inject_hfield_data(self.model)
        try:
            # 针对不同版本的兼容性刷新
            if hasattr(self, 'con'):
                self.con.free_hfield(0)
                self.con.upload_hfield(0)
        except:
            pass

    def inject_hfield_data(self, model):
        hfield_id = 0
        adr = model.hfield_adr[hfield_id]
        nrow = model.hfield_nrow[hfield_id]
        ncol = model.hfield_ncol[hfield_id]

        # 1. 生成坐标网格
        # 我们只关心 Y 方向的频率
        y_wave_frequency = 3 * np.pi
        y = np.linspace(0, y_wave_frequency, ncol)
        x = np.linspace(0, 1, nrow)  # X 方向只需线性分布，不需要周期
        X, Y = np.meshgrid(x, y)

        # 2. 核心修改：Z 只受 Y 的影响
        # 这样在 X 方向（nrow 方向）上的每一排数据都是完全一样的
        Z = (np.sin(Y)) / 2.0 + 0.5

        model.hfield_data[adr: adr + nrow * ncol] = Z.flatten()
        print("单向（Y方向）波浪地形数据已生成")

    def runBefore(self):
        self.inject_hfield_data(self.model)
        self.initial_pos = self.model.key_qpos[0]
        self.data.ctrl[:6] = self.initial_pos[:6]

        self.ee_body_name = "wrist_3_link"
        self.arm = pinocchio_kinematic.Kinematics("wrist_3_link")
        self.arm.buildFromMJCF(self.arm_xml)

        self.last_dof = self.data.ctrl[:6].copy()
        self.setTimestep(0.005)

        # 导纳状态变量：全部在向量空间/切空间定义
        self.delta_d_ee_des = np.zeros(6)  # 速度增量 [v, omega]
        self.first_goto_initial_pos_cnt = 100

        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)

        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("vel.y", "vel.y", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)
        
        # 添加力传感器绘图器
        self.plot_manager.addNewFigurePlotter("force", title="力传感器 - 力", row=0, col=1)
        self.plot_manager.addNewFigurePlotter("torque", title="力传感器 - 力矩", row=1, col=1)
        
        # 为力传感器数据添加带颜色的曲线
        self.plot_manager.addPlotToPlotter("force", "force.x", color=(255, 0, 0))  # 红色
        self.plot_manager.addPlotToPlotter("force", "force.y", color=(0, 255, 0))  # 绿色
        self.plot_manager.addPlotToPlotter("force", "force.z", color=(0, 0, 255))  # 蓝色
        self.plot_manager.addPlotToPlotter("torque", "torque.x", color=(255, 0, 0))  # 红色
        self.plot_manager.addPlotToPlotter("torque", "torque.y", color=(0, 255, 0))  # 绿色
        self.plot_manager.addPlotToPlotter("torque", "torque.z", color=(0, 0, 255))  # 蓝色
        
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
        self.initial_pos = self.model.key_qpos[0]

    def get_ee_pose_matrix(self):
        """直接从 MuJoCo 获取 4x4 变换矩阵，避开欧拉角"""
        pos = self.data.body(self.ee_body_name).xpos.copy()
        mat = self.data.body(self.ee_body_name).xmat.reshape(3, 3).copy()
        tf = np.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        return tf

    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            # 强行保持初始位置
            self.data.qpos[:6] = self.initial_pos[:6]
            self.data.ctrl[:6] = self.initial_pos[:6]

            if self.first_goto_initial_pos_cnt == 0:
                mujoco.mj_forward(self.model, self.data)
                # 记录初始参考位姿（矩阵形式）
                initial_tf = self.get_ee_pose_matrix()
                self.start_ee_pos = initial_tf[:3, 3]
                self.start_ee_rot = R.from_matrix(initial_tf[:3, :3])

                # 初始化期望位姿
                self.desired_pos = self.start_ee_pos.copy()
                self.v_y_nominal = -0.03
                self.desired_rot = self.start_ee_rot  # R 对象

                self.last_ee_pos = self.start_ee_pos.copy()
                self.last_ee_rot = self.start_ee_rot
                self.last_dof = self.initial_pos[:6]
        else:
            # 1. 获取当前状态
            curr_tf = self.get_ee_pose_matrix()
            now_pos = curr_tf[:3, 3]
            now_rot = R.from_matrix(curr_tf[:3, :3])

            # 2. 计算速度 (位置速度 + 旋转向量速度)
            v_pos = (now_pos - self.last_ee_pos) / self.model.opt.timestep
            # 旋转速度：log(R_now * R_last^T) / dt
            diff_rot_obj = now_rot * self.last_ee_rot.inv()
            v_rot = diff_rot_obj.as_rotvec() / self.model.opt.timestep
            
            now_ee_vel = np.concatenate([v_pos, v_rot])
            self.last_ee_pos = now_pos.copy()
            self.last_ee_rot = now_rot

            now_ee_vel_filter = self.vel_filter.update(now_ee_vel)

            # 3. 读取力传感器数据（需要在导纳控制之前读取）
            # 力传感器返回 6 维向量：[fx, fy, fz, tx, ty, tz]
            # 确保先更新传感器数据
            mujoco.mj_forward(self.model, self.data)
            
            force = np.zeros(3)
            torque = np.zeros(3)
            
            if self.force_sensor_id >= 0 and self.force_sensor_id < self.model.nsensor:
                sensor_addr = self.model.sensor_adr[self.force_sensor_id]
                sensor_dim = self.model.sensor_dim[self.force_sensor_id]
                
                # 确保传感器数据维度正确
                if sensor_dim >= 6 and sensor_addr + sensor_dim <= len(self.data.sensordata):
                    sensor_data = self.data.sensordata[sensor_addr:sensor_addr + sensor_dim]
                    force = sensor_data[:3].copy()  # 前3个是力
                    torque = sensor_data[3:6].copy()  # 后3个是力矩
                elif sensor_dim >= 3 and sensor_addr + sensor_dim <= len(self.data.sensordata):
                    # 如果只有3维，可能只有力没有力矩
                    sensor_data = self.data.sensordata[sensor_addr:sensor_addr + sensor_dim]
                    force = sensor_data[:3].copy()
                    torque = np.zeros(3)
                # else: 保持默认的零值

            # 4. 计算偏差 (Error in SE(3))
            err_pos = now_pos - self.desired_pos
            # 旋转偏差：目标到当前的旋转向量
            err_rot_obj = now_rot * self.desired_rot.inv()
            err_rot = err_rot_obj.as_rotvec()
            
            ee_pos_err = np.concatenate([err_pos, err_rot])

            # 5. 导纳动力学方程
            F_target = np.array([0, 0, -50, 0, 0, 0])  # 施加虚拟力
            current_f_sensor = np.zeros(6)
            current_f_sensor[2] = force[2]
            f_error =  F_target - current_f_sensor
            

            
            self.M_d = np.diag([10] * 6)
            self.B_d = np.diag([1000] * 6)
            self.K_d = np.diag([0] * 6)

            dd_ee = np.linalg.inv(self.M_d) @ (f_error - self.B_d @ now_ee_vel_filter - self.K_d @ ee_pos_err)
            print(dd_ee)

            # 6. 积分更新期望位姿
            # 累积速度：delta_d_ee_des 是速度 [v, omega]
            self.delta_d_ee_des = dd_ee * self.model.opt.timestep
            delta_step_admittance = self.delta_d_ee_des * self.model.opt.timestep
            delta_step_nominal=np.zeros(3)
            delta_step_nominal[1] = self.v_y_nominal*self.model.opt.timestep
            

            # 更新期望位置 (线性加法)
            self.desired_pos += delta_step_nominal+delta_step_admittance[:3]
            # 更新期望旋转 (旋转增量乘法)
            delta_rot_step = R.from_rotvec(delta_step_admittance[3:])
            self.desired_rot = delta_rot_step * self.desired_rot

            # 7. 组合为变换矩阵供 IK 解算
            target_tf = np.eye(4)
            target_tf[:3, :3] = self.desired_rot.as_matrix()
            target_tf[:3, 3] = self.desired_pos

            self.dof, info = self.arm.ik(target_tf, current_arm_motor_q=self.last_dof)

            # 8. 安全检查与执行
            if not self.ik_stop and info["success"]:
                self.last_dof = self.dof[:6]
                self.data.ctrl[:6] = self.dof[:6]

                # 绘图调试
                self.plot_manager.updateDataToPlotter("vel.y", "now_ee_vel.y", now_ee_vel[0])
                self.plot_manager.updateDataToPlotter("delta.x", "delta.x", self.desired_pos[0])
                self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
                self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
                
                # 绘制力传感器数据
                self.plot_manager.updateDataToPlotter("force", "force.x", force[0])
                self.plot_manager.updateDataToPlotter("force", "force.y", force[1])
                self.plot_manager.updateDataToPlotter("force", "force.z", force[2])
                self.plot_manager.updateDataToPlotter("torque", "torque.x", torque[0])
                self.plot_manager.updateDataToPlotter("torque", "torque.y", torque[1])
                self.plot_manager.updateDataToPlotter("torque", "torque.z", torque[2])
            else:
                self.data.ctrl[:6] = self.last_dof[:6]


if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene_45.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e_2.xml"
    env = Ur5eEnv(SCENE_XML, ARM_XML)
    env.run_loop()