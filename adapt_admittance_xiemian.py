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


    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]
        self.data.ctrl[:6] = self.initial_pos[:6]

        self.ee_body_name = "wrist_3_link"
        self.arm = pinocchio_kinematic.Kinematics("wrist_3_link")
        self.arm.buildFromMJCF(self.arm_xml)

        self.last_dof = self.data.ctrl[:6].copy()
        self.setTimestep(0.005)

        # 导纳状态变量
        self.delta_d_ee_des = np.zeros(6)  # 速度增量 [v, omega]
        self.first_goto_initial_pos_cnt = 100
        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)

        # ===================== 自适应导纳控制核心参数（基于论文5.3-5.4节）=====================
        self.fd_z = -50.0  # Z轴期望接触力（与论文一致，向下为负）
        self.phi = 0.0  # 自适应调整项（补偿环境位置误差，标量→Z轴专属）
        self.sigma = 0.0  # 动态更新率（随接触力变化）

        # 动态更新率参数（论文式5-39）：sigma(η) = 1/(e^(k1*(η - c0)) + k2)
        self.k1 = 3.0  # 曲线斜率（控制sigma变化速率）
        self.k2 = 100.0  # 曲线幅值（控制sigma最大值）
        self.c0 = -8.0  # 中心偏移（适配接触力范围）

        # 导纳核心参数（Z轴单独配置，符合论文单自由度力控制逻辑）
        self.md = 1.0  # 期望惯性（论文md）
        self.bd = 1000.0  # 期望阻尼（论文bd）
        self.kd = 0.0  # 期望刚度（论文kd=0，避免静态误差）

        # 稳定性约束参数（论文式5-16：0 < sigma < bd*Ts/(md + bd*Ts)）
        self.Ts = self.model.opt.timestep  # 控制周期
        self.sigma_max = (self.bd * self.Ts) / (self.md + self.bd * self.Ts)

        # 历史数据缓存（用于自适应调整项积分更新）
        self.last_force_error = 0.0
        # =====================================================================================

        # 绘图配置
        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("vel.y", "vel.y", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)

        # 力传感器绘图（新增自适应参数曲线）
        self.plot_manager.addNewFigurePlotter("force", title="力传感器 - 力", row=0, col=1)
        self.plot_manager.addNewFigurePlotter("torque", title="力传感器 - 力矩", row=1, col=1)
        self.plot_manager.addNewFigurePlotter("adaptive", title="自适应参数（phi/sigma）", row=2, col=1)

        # 为力传感器数据添加颜色曲线
        self.plot_manager.addPlotToPlotter("force", "force.x", color=(255, 0, 0))
        self.plot_manager.addPlotToPlotter("force", "force.y", color=(0, 255, 0))
        self.plot_manager.addPlotToPlotter("force", "force.z", color=(0, 0, 255))
        self.plot_manager.addPlotToPlotter("torque", "torque.x", color=(255, 0, 0))
        self.plot_manager.addPlotToPlotter("torque", "torque.y", color=(0, 255, 0))
        self.plot_manager.addPlotToPlotter("torque", "torque.z", color=(0, 0, 255))
        self.plot_manager.addPlotToPlotter("adaptive", "phi", color=(255, 165, 0))  # 橙色：自适应调整项
        self.plot_manager.addPlotToPlotter("adaptive", "sigma", color=(128, 0, 128))  # 紫色：动态更新率

        # 获取力传感器ID
        mujoco.mj_forward(self.model, self.data)
        self.force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor_site")
        if self.force_sensor_id < 0:
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
                    self.force_sensor_id = i
                    break
        self.ik_stop = False
        self.initial_pos = self.model.key_qpos[0]

    def get_ee_pose_matrix(self):
        """直接从 MuJoCo 获取 4x4 变换矩阵"""
        pos = self.data.body(self.ee_body_name).xpos.copy()
        mat = self.data.body(self.ee_body_name).xmat.reshape(3, 3).copy()
        tf = np.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        return tf

    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            self.data.qpos[:6] = self.initial_pos[:6]
            self.data.ctrl[:6] = self.initial_pos[:6]

            if self.first_goto_initial_pos_cnt == 0:
                mujoco.mj_forward(self.model, self.data)
                initial_tf = self.get_ee_pose_matrix()
                self.start_ee_pos = initial_tf[:3, 3]
                self.start_ee_rot = R.from_matrix(initial_tf[:3, :3])

                # 初始化期望位姿
                self.desired_pos = self.start_ee_pos.copy()
                self.v_y_nominal = -0.03  # Y轴名义速度
                self.desired_rot = self.start_ee_rot

                self.last_ee_pos = self.start_ee_pos.copy()
                self.last_ee_rot = self.start_ee_rot
                self.last_dof = self.initial_pos[:6]
        else:
            # 1. 获取当前末端位姿和速度
            curr_tf = self.get_ee_pose_matrix()
            now_pos = curr_tf[:3, 3]
            now_rot = R.from_matrix(curr_tf[:3, :3])

            # 计算速度（位置速度 + 旋转速度）
            v_pos = (now_pos - self.last_ee_pos) / self.Ts
            diff_rot_obj = now_rot * self.last_ee_rot.inv()
            v_rot = diff_rot_obj.as_rotvec() / self.Ts
            now_ee_vel = np.concatenate([v_pos, v_rot])
            self.last_ee_pos = now_pos.copy()
            self.last_ee_rot = now_rot
            now_ee_vel_filter = self.vel_filter.update(now_ee_vel)

            # 2. 读取Z轴接触力（论文中力控制核心方向）
            mujoco.mj_forward(self.model, self.data)
            fe_z = 0.0  # 实际接触力（Z轴）
            if self.force_sensor_id >= 0 and self.force_sensor_id < self.model.nsensor:
                sensor_addr = self.model.sensor_adr[self.force_sensor_id]
                sensor_dim = self.model.sensor_dim[self.force_sensor_id]
                if sensor_addr + sensor_dim <= len(self.data.sensordata):
                    sensor_data = self.data.sensordata[sensor_addr:sensor_addr + sensor_dim]
                    fe_z = sensor_data[2] if sensor_dim >= 3 else 0.0  # 提取Z轴力

            # ===================== 自适应导纳核心逻辑（论文5.3-5.4节）=====================
            # 步骤1：计算动态更新率sigma(η)（η=接触力fe_z）
            eta = fe_z  # 输入为Z轴接触力（与论文一致）
            exponent = self.k1 * (eta - self.c0)
            self.sigma = 1.0 / (np.exp(exponent) + self.k2)
            # 限制sigma在稳定范围内（论文式5-16）
            self.sigma = np.clip(self.sigma, 1e-6, self.sigma_max)

            # 步骤2：计算力误差（论文式5-15：fd - fe）
            current_force_error = self.fd_z - fe_z

            # 步骤3：更新自适应调整项φ(t)（积分补偿环境误差）
            # 论文式5-15：φ(t) = φ(t-Ts) + sigma*(fd(t-Ts) - fe(t-Ts))/bd
            self.phi += self.sigma * current_force_error / self.bd
            # 限制φ幅值，避免过度补偿导致振荡
            self.phi = np.clip(self.phi, -0.1, 0.1)

            # 步骤4：自适应导纳动力学方程（转换为导纳控制形式）
            # 论文阻抗方程：fe - fd = md*ddot(êx) + bd*(dot(êx) + φ)
            # 转换为导纳（力→加速度）：ddot(z) = (fd - fe - bd*φ - bd*dot(z))/md
            F_error_z = current_force_error
            #F_error_z = current_force_error - self.bd * self.phi  # 自适应力误差补偿
            dd_ee_z = (F_error_z - self.bd * now_ee_vel_filter[2] - self.kd * (
                        now_pos[2] - self.desired_pos[2])) / self.md
            # =====================================================================================

            # 3. 构建6维加速度（仅Z轴自适应，其他维度保持原导纳）
            dd_ee = np.zeros(6)
            # 其他维度保持原导纳控制
            for i in range(6):
                if i != 2:
                    dd_ee[i] = 0
                else:
                    dd_ee[i] = dd_ee_z  # Z轴使用自适应导纳结果

            # 4. 积分更新期望位姿
            self.delta_d_ee_des = dd_ee * self.Ts
            delta_step_admittance = self.delta_d_ee_des * self.Ts
            delta_step_nominal = np.zeros(3)
            delta_step_nominal[1] = self.v_y_nominal * self.Ts

            # 更新期望位置和姿态
            self.desired_pos += delta_step_nominal + delta_step_admittance[:3]
            delta_rot_step = R.from_rotvec(delta_step_admittance[3:])
            self.desired_rot = delta_rot_step * self.desired_rot

            # 5. IK解算与控制执行
            target_tf = np.eye(4)
            target_tf[:3, :3] = self.desired_rot.as_matrix()
            target_tf[:3, 3] = self.desired_pos
            self.dof, info = self.arm.ik(target_tf, current_arm_motor_q=self.last_dof)

            if not self.ik_stop and info["success"]:
                self.last_dof = self.dof[:6]
                self.data.ctrl[:6] = self.dof[:6]

                # 更新绘图数据（新增自适应参数）
                self.plot_manager.updateDataToPlotter("vel.y", "now_ee_vel.y", now_ee_vel[0])
                self.plot_manager.updateDataToPlotter("delta.x", "delta.x", self.desired_pos[0])
                self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
                self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
                self.plot_manager.updateDataToPlotter("force", "force.x",
                                                      sensor_data[0] if len(sensor_data) >= 3 else 0.0)
                self.plot_manager.updateDataToPlotter("force", "force.y",
                                                      sensor_data[1] if len(sensor_data) >= 3 else 0.0)
                self.plot_manager.updateDataToPlotter("force", "force.z", fe_z)
                self.plot_manager.updateDataToPlotter("torque", "torque.x",
                                                      sensor_data[3] if len(sensor_data) >= 6 else 0.0)
                self.plot_manager.updateDataToPlotter("torque", "torque.y",
                                                      sensor_data[4] if len(sensor_data) >= 6 else 0.0)
                self.plot_manager.updateDataToPlotter("torque", "torque.z",
                                                      sensor_data[5] if len(sensor_data) >= 6 else 0.0)
                self.plot_manager.updateDataToPlotter("adaptive", "phi", self.phi)
                self.plot_manager.updateDataToPlotter("adaptive", "sigma", self.sigma)
            else:
                self.data.ctrl[:6] = self.last_dof[:6]


if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e_2.xml"
    env = Ur5eEnv(SCENE_XML, ARM_XML)
    env.run_loop()