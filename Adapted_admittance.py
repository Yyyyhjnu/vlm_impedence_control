import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.mujoco_viewer as mujoco_viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import src.lowpass_filter as lowpass_filter
import PyKDL
import math


# =============================================================================================
# 1. PyKDL UR5e 运动学链定义（保留原官方 DH 参数）
# =============================================================================================
def create_ur5e_robot_chain() -> PyKDL.Chain:
    """创建 UR5e 6 自由度机械臂的 PyKDL 运动学链（官方 DH 参数）"""
    ur5e_chain = PyKDL.Chain()

    # 关节1: 基座 → 肩关节
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(0.0, math.pi/2, 0.1625, 0.0)
        )
    )

    # 关节2: 肩关节 → 肘关节
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(-0.425, 0.0, 0.0, 0.0)
        )
    )

    # 关节3: 肘关节 → 腕关节1
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(-0.3922, 0.0, 0.0, 0.0)
        )
    )

    # 关节4: 腕关节1 → 腕关节2
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(0.0, math.pi/2, 0.1333, 0.0)
        )
    )

    # 关节5: 腕关节2 → 腕关节3
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(0.0, -math.pi/2, 0.0997, 0.0)
        )
    )

    # 关节6: 腕关节3 → 末端法兰
    ur5e_chain.addSegment(
        PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame.DH(0.0, 0.0, 0.0996, 0.0)
        )
    )

    return ur5e_chain


# =============================================================================================
# 2. 主环境类（用 PyKDL 替换 Pinocchio 运动学）
# =============================================================================================
class Ur5eEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.arm_xml = arm_xml

        # 数据保存初始化
        self.save_data_list = []
        self.start_timestamp = time.time()
        self.save_dir = "./ur5e_adaptive_data"
        os.makedirs(self.save_dir, exist_ok=True)
        self.fd_z = 50.0

    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]
        self.data.ctrl[:6] = self.initial_pos[:6]

        self.ee_body_name = "wrist_3_link"
        self.last_dof = self.data.ctrl[:6].copy()
        self.setTimestep(0.005)

        # 导纳状态变量
        self.delta_d_ee_des = np.zeros(6)
        self.first_goto_initial_pos_cnt = 100
        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)

        # ===================== PyKDL 运动学求解器初始化（核心替换）=====================
        self.ur5e_chain = create_ur5e_robot_chain()
        # 正运动学求解器（递归法）
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.ur5e_chain)
        # 速度逆运动学求解器（伪逆法）
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.ur5e_chain)
        # 位置逆运动学求解器（牛顿-拉夫逊法，依赖正运动学和速度逆运动学）
        self.ik_solver = PyKDL.ChainIkSolverPos_NR(
            self.ur5e_chain,
            self.fk_solver,
            self.ik_vel_solver,
            100,  # 最大迭代次数
            1e-6  # 收敛阈值
        )
        # =================================================================================

        # 自适应导纳控制核心参数
        self.phi = 0.0
        self.sigma = 0.0
        self.k1 = 3.0
        self.k2 = 100.0
        self.c0 = -8.0
        self.md = 3.0
        self.bd = 1000.0
        self.kd = 0.0
        self.Ts = self.model.opt.timestep
        self.sigma_max = (self.bd * self.Ts) / (self.md + self.bd * self.Ts)
        self.last_force_error = 0.0

        # 绘图配置
        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("force", title="力传感器 - 力（基坐标系）", row=1, col=0)
        self.plot_manager.addNewFigurePlotter("adaptive", title="自适应参数（phi/sigma）", row=2, col=0)
        self.plot_manager.addPlotToPlotter("force", "force.z", color=(0, 0, 255))
        self.plot_manager.addPlotToPlotter("adaptive", "phi", color=(255, 165, 0))
        self.plot_manager.addPlotToPlotter("adaptive", "sigma", color=(128, 0, 128))

        # 力传感器配置
        mujoco.mj_forward(self.model, self.data)
        self.force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor_site")
        if self.force_sensor_id < 0:
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
                    self.force_sensor_id = i
                    break
        self.sensor_ref_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

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

    def transform_force_position_to_base_frame(self, force_sensor, torque_sensor):
        """将传感器坐标系下的力/力矩转换到基坐标系"""
        if self.sensor_ref_body_id < 0:
            return force_sensor.copy(), torque_sensor.copy()
        try:
            body_rot_mat = self.data.xmat[self.sensor_ref_body_id].reshape(3, 3).copy()
            force_base = np.dot(body_rot_mat, force_sensor)
            torque_base = np.dot(body_rot_mat, torque_sensor)
            return force_base, torque_base
        except Exception as e:
            print(f"⚠️ 坐标转换出错：{e}")
            return force_sensor.copy(), torque_sensor.copy()

    def plot_collected_data(self, df, save_path):
        """根据保存的DataFrame绘制可视化图表（与原代码一致）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('UR5e自适应导纳控制实验数据（基坐标系）', fontsize=16, fontweight='bold')

        time_data = df['time_s']
        y_pos_data = df['y_pos_m']
        z_pos_data = df['z_pos_m']
        fz_base_data = df['fz_force_base_N']
        phi_data = df['adaptive_phi']
        sigma_data = df['dynamic_sigma']

        # 子图1：Y轴位置
        ax1 = axes[0, 0]
        ax1.plot(time_data, y_pos_data, color='#2E86AB', linewidth=1.5, label='Y轴位置')
        ax1.set_title('Y轴末端位置时序曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('位置 (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 子图2：Z轴位置
        ax2 = axes[0, 1]
        ax2.plot(time_data, z_pos_data, color='#A23B72', linewidth=1.5, label='Z轴位置')
        ax2.set_title('Z轴末端位置时序曲线', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('位置 (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 子图3：Z轴接触力
        ax3 = axes[1, 0]
        ax3.plot(time_data, fz_base_data, color='#F18F01', linewidth=1.5, label='实际接触力（基坐标系）')
        ax3.axhline(y=self.fd_z, color='#C73E1D', linestyle='--', linewidth=1.5, label=f'期望接触力({self.fd_z}N)')
        ax3.set_title('Z轴接触力时序曲线（基坐标系）', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('力 (N)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 子图4：自适应参数
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(time_data, phi_data, color='#6A994E', linewidth=1.5, label='自适应调整项φ')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('自适应调整项 φ', color='#6A994E')
        ax4.tick_params(axis='y', labelcolor='#6A994E')
        ax4.grid(True, alpha=0.3)
        line2 = ax4_twin.plot(time_data, sigma_data, color='#BC4749', linewidth=1.5, label='动态更新率σ')
        ax4_twin.set_ylabel('动态更新率 σ', color='#BC4749')
        ax4_twin.tick_params(axis='y', labelcolor='#BC4749')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        ax4.set_title('自适应参数时序曲线', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_save_path = save_path.replace('.csv', '.png')
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 实验数据图表已保存至：{plot_save_path}")

    def save_collected_data(self):
        """将缓存的实验数据保存为CSV文件"""
        if not self.save_data_list:
            print("⚠️ 无数据可保存！")
            return
        columns = [
            "time_s", "y_pos_m", "z_pos_m",
            "fz_force_sensor_N", "fz_force_base_N",
            "adaptive_phi", "dynamic_sigma"
        ]
        df = pd.DataFrame(self.save_data_list, columns=columns)
        current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path = os.path.join(self.save_dir, f"ur5e_adaptive_data_{current_time_str}.csv")
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ 数据已保存至：{save_path}")
        self.plot_collected_data(df, save_path)

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
                self.desired_pos = self.start_ee_pos.copy()
                self.v_y_nominal = -0.05
                self.desired_rot = self.start_ee_rot
                self.last_ee_pos = self.start_ee_pos.copy()
                self.last_ee_rot = self.start_ee_rot
                self.last_dof = self.initial_pos[:6]
        else:
            # 1. 获取当前末端位姿和速度
            curr_tf = self.get_ee_pose_matrix()
            now_pos = curr_tf[:3, 3]
            now_rot = R.from_matrix(curr_tf[:3, :3])

            # 计算速度（补全原代码缺失的 v_rot）
            v_pos = (now_pos - self.last_ee_pos) / self.Ts
            diff_rot_obj = now_rot * self.last_ee_rot.inv()
            v_rot = diff_rot_obj.as_rotvec() / self.Ts  # 旋转速度（旋转向量差分除以时间）
            now_ee_vel = np.concatenate([v_pos, v_rot])
            self.last_ee_pos = now_pos.copy()
            self.last_ee_rot = now_rot

            # 2. 读取力传感器数据
            mujoco.mj_forward(self.model, self.data)
            force_sensor = np.zeros(3)
            torque_sensor = np.zeros(3)
            fe_z_sensor = 0.0
            sensor_addr = self.model.sensor_adr[self.force_sensor_id]
            sensor_dim = self.model.sensor_dim[self.force_sensor_id]
            if sensor_addr + sensor_dim <= len(self.data.sensordata):
                sensor_data = self.data.sensordata[sensor_addr:sensor_addr + sensor_dim]
                if sensor_dim == 3:
                    force_sensor = sensor_data[:3].copy()
                    fe_z_sensor = force_sensor[2]
                if sensor_dim == 6:
                    torque_sensor = sensor_data[3:6].copy()

            # 转换力到基坐标系
            force_base, torque_base = self.transform_force_position_to_base_frame(force_sensor, torque_sensor)
            fe_z_base = force_base[2]

            # ===================== 自适应导纳核心逻辑（与原代码一致）=====================
            eta = fe_z_base
            exponent = self.k1 * (eta - self.c0)
            self.sigma = 0.001
            print(f"sigma: {self.sigma:.6f}, fe_z_base: {fe_z_base:.3f}")

            current_force_error = fe_z_base - self.fd_z
            self.phi += self.sigma * current_force_error / self.bd
            self.phi = np.clip(self.phi, -0.1, 0.1)

            F_error_z = current_force_error
            dd_ee_z = (F_error_z - self.bd * (v_pos[2] + self.phi) - self.kd * (
                    now_pos[2] - self.desired_pos[2])) / self.md
            # =================================================================================

            # 3. 构建6维加速度
            dd_ee = np.zeros(6)
            for i in range(6):
                if i != 2:
                    dd_ee[i] = 0
                else:
                    dd_ee[i] = dd_ee_z

            # 4. 积分更新期望位姿
            self.delta_d_ee_des = dd_ee * self.Ts
            delta_step_admittance = self.delta_d_ee_des * self.Ts
            delta_step_nominal = np.zeros(3)
            delta_step_nominal[1] = self.v_y_nominal * self.Ts

            self.desired_pos += delta_step_nominal + delta_step_admittance[:3]
            delta_rot_step = R.from_rotvec(delta_step_admittance[3:])
            self.desired_rot = delta_rot_step * self.desired_rot

            # ===================== PyKDL 逆运动学求解（核心替换）=====================
            # 将 numpy 目标位姿转换为 PyKDL Frame
            target_frame = PyKDL.Frame(
                PyKDL.Rotation(self.desired_rot.as_matrix()[0,0], self.desired_rot.as_matrix()[0,1], self.desired_rot.as_matrix()[0,2],
                               self.desired_rot.as_matrix()[1,0], self.desired_rot.as_matrix()[1,1], self.desired_rot.as_matrix()[1,2],
                               self.desired_rot.as_matrix()[2,0], self.desired_rot.as_matrix()[2,1], self.desired_rot.as_matrix()[2,2]),
                PyKDL.Vector(self.desired_pos[0], self.desired_pos[1], self.desired_pos[2])
            )

            # 将 numpy 初始猜测转换为 PyKDL JntArray
            q_init = PyKDL.JntArray(6)
            for i in range(6):
                q_init[i] = self.last_dof[i]

            # 求解逆运动学
            q_result = PyKDL.JntArray(6)
            ik_success = (self.ik_solver.CartToJnt(q_init, target_frame, q_result) >= 0)
            # =========================================================================

            if not self.ik_stop and ik_success:
                # 将 PyKDL JntArray 转换回 numpy
                self.dof = np.zeros(6)
                for i in range(6):
                    self.dof[i] = q_result[i]

                self.last_dof = self.dof[:6]
                self.data.ctrl[:6] = self.dof[:6]

                # 更新绘图数据
                self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
                self.plot_manager.updateDataToPlotter("force", "force.z", force_base[2])
                self.plot_manager.updateDataToPlotter("adaptive", "phi", self.phi)
                self.plot_manager.updateDataToPlotter("adaptive", "sigma", self.sigma)

                # 缓存待保存的数据
                current_time = time.time() - self.start_timestamp
                self.save_data_list.append([
                    round(current_time, 4),
                    round(self.desired_pos[1], 6),
                    round(self.desired_pos[2], 6),
                    round(fe_z_sensor, 3),
                    round(fe_z_base, 3),
                    round(self.phi, 6),
                    round(self.sigma, 6)
                ])
            else:
                self.data.ctrl[:6] = self.last_dof[:6]


if __name__ == "__main__":
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    ARM_XML = "model/universal_robots_ur5e/ur5e_2.xml"
    env = Ur5eEnv(SCENE_XML, ARM_XML)
    env.run_loop()
    env.save_collected_data()