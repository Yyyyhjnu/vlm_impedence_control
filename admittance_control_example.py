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

class ur5e_Env(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml

    def runBefore(self):
        # 1. 获取初始角度（6位）
        self.initial_q = self.model.key_qpos[0]

        # 2. 初始化 ctrl (长度为12)
        # 前6位是位置执行器，设为初始姿态；后6位是力矩执行器，设为0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[:6] = self.initial_q

        self.ee_body_name = "force_sensor"
        self.arm = pinocchio_kinematic.Kinematics("force_sensor")
        self.arm.buildFromMJCF(self.arm_xml)

        # 确保 last_dof 也是6位
        self.last_dof = self.data.qpos[:6].copy()
        self.setTimestep(0.001)

        # 状态初始化
        self.delta_d_ee_des = np.zeros(6)
        self.delta_ee_des = np.zeros(6)
        self.first_goto_initial_pos_cnt = 100

        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        self.acc_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("vel.x", "vel.x", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)
        self.ik_stop = False

    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            self.data.ctrl[:6] = self.initial_q[:6]
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.desired_pos = self.ee_pos
            self.last_ee_pos = self.ee_pos
            self.start_ee_pos = self.ee_pos
        else:
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.now_ee_vel = (self.now_ee_pos - self.last_ee_pos) / self.model.opt.timestep
            self.last_ee_pos = self.now_ee_pos
            self.now_ee_vel_filter = self.vel_filter.update(self.now_ee_vel)
            ee_pos_err = self.now_ee_pos - self.desired_pos

            F_ref = np.zeros(6)
            F_meas = np.array([0, 0, 0, 0, 0, 0])
            self.axis_index = 1
            F_meas[self.axis_index] = 1
            F_e = F_meas - F_ref

            self.M_d = np.diag([10] * 6)
            self.B_d = np.diag([0] * 6)
            self.K_d = np.diag([50] * 6)
            # M_d·ddq + B_d·dq + K_d·(q_des - q) = F_e
            dd_ee = np.linalg.inv(self.M_d) @ (F_e - self.B_d @ self.now_ee_vel - self.K_d @ ee_pos_err)

            # 检查计算值是否有效
            if np.any(np.isnan(dd_ee)) or np.any(np.isinf(dd_ee)):
                print("警告: 检测到 NaN 或 Inf 值，跳过本次更新")
                return
            
            self.delta_d_ee_des += dd_ee * self.model.opt.timestep
            self.delta_ee_des += self.delta_d_ee_des * self.model.opt.timestep
            
            # 限制位置增量，避免过大变化
            max_delta = 0.1  # 最大位置增量（米或弧度）
            self.delta_ee_des = np.clip(self.delta_ee_des, -max_delta, max_delta)
            
            self.desired_pos[:6] = self.start_ee_pos[:6] + self.delta_ee_des[:6]

            tf = utils.transform2mat(self.desired_pos[0], self.desired_pos[1], self.desired_pos[2], self.desired_pos[3],
                                     self.desired_pos[4], self.desired_pos[5])
            self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
            
            # 检查 IK 求解结果是否有效
            if not info["success"] or np.any(np.isnan(self.dof)) or np.any(np.isinf(self.dof)):
                print(f"警告: IK 求解失败或结果无效，保持上一状态")
                self.ik_stop = True
                return
            
            # 检查关节角度是否在合理范围内
            joint_limits = np.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28])  # ±π 弧度
            if np.any(np.abs(self.dof[:6]) > joint_limits):
                print(f"警告: 关节角度超出限制，停止控制")
                self.ik_stop = True
                return
            
            if self.desired_pos[self.axis_index] < 0.001:
                self.ik_stop = True
            
            if not self.ik_stop:
                self.last_dof = self.dof.copy()
                # 使用控制器设置期望位置，而不是直接设置 qpos（避免位置不连续）
                self.data.ctrl[:6] = self.dof[:6]
                self.plot_manager.updateDataToPlotter("vel.x", "now_ee_vel.x", self.now_ee_vel[0])
                self.plot_manager.updateDataToPlotter("vel.x", "now_ee_velfilter.x", self.now_ee_vel_filter[0])
                self.plot_manager.updateDataToPlotter("delta.x", "delta.x", self.desired_pos[0])
                self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
                self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
            else:
                # 停止时也使用控制器保持位置
                self.data.ctrl[:6] = self.last_dof[:6]

if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e.xml"
    env = ur5e_Env(SCENE_XML, ARM_XML)
    env.run_loop()