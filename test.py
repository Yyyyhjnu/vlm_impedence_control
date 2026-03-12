#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR5e 末端位置调试工具（PyKDL + MuJoCo 可视化）
核心规则：
✅ 逆运动学初始猜测值仅从关节限位范围内选取
✅ 求解成功判定：关节合规 + 末端位置误差在允许范围内
✅ 无角度裁剪，超出限位/误差超标的结果直接舍弃
✅ 重试失败则提示无解，无任何兜底操作
"""
"""
import PyKDL
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer as mj_viewer
import time
import threading
import random


# =============================================================================================
# 1. 核心：UR5e PyKDL 运动学链（官方 DH 参数）
# =============================================================================================
def create_ur5e_chain() -> PyKDL.Chain:
    #创建 UR5e 6 自由度机械臂的 PyKDL 运动学链
    ur5e_chain = PyKDL.Chain()

    # 关节1: 基座 → 肩关节
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(0.0, math.pi / 2, 0.1625, 0.0)
    ))

    # 关节2: 肩关节 → 肘关节
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(-0.425, 0.0, 0.0, 0.0)
    ))

    # 关节3: 肘关节 → 腕关节1
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(-0.3922, 0.0, 0.0, 0.0)
    ))

    # 关节4: 腕关节1 → 腕关节2
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(0.0, math.pi / 2, 0.1333, 0.0)
    ))

    # 关节5: 腕关节2 → 腕关节3
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(0.0, -math.pi / 2, 0.0997, 0.0)
    ))

    # 关节6: 腕关节3 → 末端法兰
    ur5e_chain.addSegment(PyKDL.Segment(
        PyKDL.Joint(PyKDL.Joint.RotZ),
        PyKDL.Frame.DH(0.0, 0.0, 0.0996, 0.0)
    ))

    return ur5e_chain


# =============================================================================================
# 2. 工具函数：PyKDL ↔ Numpy 转换
# =============================================================================================
def jntarray_to_numpy(jnt_array: PyKDL.JntArray) -> np.ndarray:
    #PyKDL.JntArray → numpy数组
    return np.array([jnt_array[i] for i in range(jnt_array.rows())])


def numpy_to_jntarray(numpy_array: np.ndarray) -> PyKDL.JntArray:
    #numpy数组 → PyKDL.JntArray
    jnt_array = PyKDL.JntArray(len(numpy_array))
    for i in range(len(numpy_array)):
        jnt_array[i] = numpy_array[i]
    return jnt_array


def frame_to_numpy(frame: PyKDL.Frame) -> tuple[np.ndarray, np.ndarray]:
    PyKDL.Frame → 位置+旋转矩阵
    pos = np.array([frame.p.x(), frame.p.y(), frame.p.z()])
    rot_mat = np.array([
        [frame.M[0, 0], frame.M[0, 1], frame.M[0, 2]],
        [frame.M[1, 0], frame.M[1, 1], frame.M[1, 2]],
        [frame.M[2, 0], frame.M[2, 1], frame.M[2, 2]]
    ])
    return pos, rot_mat


def numpy_to_frame(pos: np.ndarray, rot_mat: np.ndarray) -> PyKDL.Frame:
    位置+旋转矩阵 → PyKDL.Frame
    return PyKDL.Frame(
        PyKDL.Rotation(
            rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2],
            rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2],
            rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]
        ),
        PyKDL.Vector(pos[0], pos[1], pos[2])
    )


# =============================================================================================
# 3. 正/逆运动学求解
# =============================================================================================
def forward_kinematics(chain: PyKDL.Chain, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    正运动学：关节角度 → 末端位姿
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    q = numpy_to_jntarray(joint_angles)
    end_frame = PyKDL.Frame()
    fk_solver.JntToCart(q, end_frame)
    return frame_to_numpy(end_frame)


def inverse_kinematics(
        chain: PyKDL.Chain,
        target_pos: np.ndarray,
        target_rot_mat: np.ndarray,
        q_init: np.ndarray = None
) -> tuple[bool, np.ndarray]:
    逆运动学：末端位姿 → 关节角度
    if q_init is None:
        q_init = np.zeros(6)

    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    ik_pos_solver = PyKDL.ChainIkSolverPos_NR(
        chain, fk_solver, ik_vel_solver,
        maxiter=200, eps=1e-6
    )

    target_frame = numpy_to_frame(target_pos, target_rot_mat)
    q_init_kdl = numpy_to_jntarray(q_init)
    q_result_kdl = PyKDL.JntArray(6)

    success = (ik_pos_solver.CartToJnt(q_init_kdl, target_frame, q_result_kdl) >= 0)
    q_result = jntarray_to_numpy(q_result_kdl)
    return success, q_result


# =============================================================================================
# 4. MuJoCo 可视化核心类（误差判定 + 限位内猜测值）
# =============================================================================================
class UR5eVisualDebugger:
    def __init__(self, mjcf_path: str):
        # 1. 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # 2. 初始化 PyKDL 运动学链
        self.ur5e_chain = create_ur5e_chain()

        # 3. 调试参数
        self.target_pos = np.array([0.0, -0.5, 0.5])  # 默认目标位置
        self.target_rot = R.identity().as_matrix()  # 默认目标姿态
        self.current_joints = np.zeros(6)  # 当前关节角度
        self.ee_body_name = "wrist_3_link"  # 末端连杆名称
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        # ===================== 核心配置项 =====================
        # 关节限位配置（仅从该范围选取初始猜测值）
        self.joint_limits = np.array([
            [math.pi, math.pi],  # 关节1：基座旋转
            [-math.pi / 2, -0.2],  # 关节2：肩关节
            [0.1, 2],  # 关节3：肘关节
            [-0.5, 0.5],  # 关节4：腕关节1
            [-math.pi, -math.pi],  # 关节5：腕关节2
            [-math.pi, math.pi]  # 关节6：腕关节3
        ])
        # 位置误差允许阈值（米）：误差小于该值才判定成功
        self.pos_error_threshold = 0.01  # 可根据需求调整（比如0.005=5mm）
        # 重试配置
        self.max_retry = 5000000000  # 逆运动学最大重试次数
        self.retry_delay = 0.001  # 重试间隔（缩短间隔提高效率）

        # 4. 控制标志
        self.running = True
        self.update_joints = False
        self.update_target = False

    def get_ee_pos_from_mujoco(self) -> np.ndarray:
        从 MuJoCo 获取当前末端位置
        return self.data.xpos[self.ee_body_id].copy()

    def set_joints_to_mujoco(self, joint_angles: np.ndarray):
        设置 MuJoCo 关节角度（仅接受合规且误差达标的角度）
        # 严格检查：仅当角度完全符合限位时才应用
        if not self.is_joint_in_limit(joint_angles):
            print("   ❌ 关节角度超出限位，拒绝应用到MuJoCo！")
            return None

        # 应用合规的角度
        self.data.qpos[:6] = joint_angles
        self.data.ctrl[:6] = joint_angles
        mujoco.mj_forward(self.model, self.data)
        return joint_angles

    def is_joint_in_limit(self, joint_angles: np.ndarray) -> bool:
        严格检查：所有关节角度是否完全在限位范围内
        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            if joint_angles[i] < min_val or joint_angles[i] > max_val:
                return False
        return True

    def calculate_pos_error(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        计算关节角度对应的末端位置与目标位置的误差（欧氏距离）
        ee_pos, _ = forward_kinematics(self.ur5e_chain, joint_angles)
        error = np.linalg.norm(ee_pos - target_pos)
        return error

    def is_pos_error_acceptable(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> bool:
        判断末端位置误差是否在允许范围内
        error = self.calculate_pos_error(joint_angles, target_pos)
        return error <= self.pos_error_threshold

    def generate_q_init_from_limit(self) -> np.ndarray:
        仅从关节限位范围内生成随机初始猜测值（核心修改）
        q_init = np.zeros(6)
        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            # 确保生成的值严格在限位内（避免边界值问题）
            q_init[i] = random.uniform(min_val + 1e-6, max_val - 1e-6)
        return q_init

    def print_joint_limits(self):
        打印当前配置（限位+误差阈值）
        print("\n📏 当前配置：")
        joint_names = ["基座旋转", "肩关节", "肘关节", "腕关节1", "腕关节2", "腕关节3"]
        for i in range(6):
            min_rad, max_rad = self.joint_limits[i]
            min_deg = math.degrees(min_rad)
            max_deg = math.degrees(max_rad)
            print(f"   关节{i + 1}（{joint_names[i]}）：{min_rad:.4f}({min_deg:.1f}°) ~ {max_rad:.4f}({max_deg:.1f}°)")
        print(f"   位置误差允许阈值：{self.pos_error_threshold:.4f} 米（{self.pos_error_threshold * 1000:.1f} 毫米）")

    def start_visualization(self):
        启动 MuJoCo 可视化界面
        # 初始化关节：仅使用限位内的初始值
        if self.is_joint_in_limit(self.current_joints):
            self.set_joints_to_mujoco(self.current_joints)
        else:
            print("⚠️  默认零位超出限位，使用限位内初始值")
            self.current_joints = self.generate_q_init_from_limit()
            self.set_joints_to_mujoco(self.current_joints)

        # 启动可视化线程
        self.viewer = mj_viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = -45  # 相机方位角
        self.viewer.cam.elevation = -30  # 相机仰角
        self.viewer.cam.distance = 1.5  # 相机距离

        # 主循环
        while self.running and self.viewer.is_running():
            step_start = time.time()

            # 更新关节角度（仅应用合规解）
            if self.update_joints:
                self.set_joints_to_mujoco(self.current_joints)
                self.update_joints = False

            # 刷新可视化
            self.viewer.sync()

            # 控制帧率
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # 关闭可视化
        self.viewer.close()

    def debug_joint_angles(self, joint_angles: np.ndarray):
        调试：输入关节角度（仅接受合规值）
        print(f"\n📌 调试关节角度：{np.round(joint_angles, 4)} 弧度")

        # 1. 严格检查限位（超出则直接舍弃）
        if not self.is_joint_in_limit(joint_angles):
            print("   ❌ 输入的关节角度超出限位，直接舍弃！")
            # 打印超出的关节
            for i in range(6):
                min_val, max_val = self.joint_limits[i]
                if joint_angles[i] < min_val or joint_angles[i] > max_val:
                    print(f"      关节{i + 1}：{joint_angles[i]:.4f} 超出 [{min_val:.4f}, {max_val:.4f}]")
            return

        # 2. 合规：计算末端位置和误差
        ee_pos_pykdl, _ = forward_kinematics(self.ur5e_chain, joint_angles)
        error = np.linalg.norm(ee_pos_pykdl - self.target_pos)
        print(f"   ✅ 关节角度合规")
        print(f"   末端位置：x={ee_pos_pykdl[0]:.4f}, y={ee_pos_pykdl[1]:.4f}, z={ee_pos_pykdl[2]:.4f} m")
        print(f"   与默认目标位置误差：{error:.6f} 米")

        # 3. 应用到 MuJoCo 并获取实际末端位置
        self.current_joints = joint_angles
        self.update_joints = True
        time.sleep(0.1)  # 等待可视化更新
        ee_pos_mujoco = self.get_ee_pos_from_mujoco()
        print(f"   MuJoCo 实际末端位置：x={ee_pos_mujoco[0]:.4f}, y={ee_pos_mujoco[1]:.4f}, z={ee_pos_mujoco[2]:.4f} m")

        # 4. 计算偏差
        pos_error = np.linalg.norm(ee_pos_pykdl - ee_pos_mujoco)
        print(f"   模型与仿真偏差：{pos_error:.6f} m（越小越好）")

    def debug_target_position(self, target_pos: np.ndarray, target_rot: np.ndarray = None):
        调试：输入目标位置（核心逻辑：限位内猜测值 + 误差判定）
        if target_rot is None:
            target_rot = R.identity().as_matrix()

        print(f"\n🎯 调试目标位置：x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f} m")
        print(f"   📏 位置误差允许阈值：{self.pos_error_threshold:.4f} 米")

        # 初始化重试参数
        retry_count = 0
        success = False
        valid_joint_angles = None
        final_error = float('inf')

        # 逆运动学重试循环（核心修改）
        while retry_count < self.max_retry and not success:
            # 1. 仅从限位范围内生成初始猜测值
            q_init = self.generate_q_init_from_limit()

            # 2. 逆运动学求解
            solve_success, joint_angles = inverse_kinematics(
                self.ur5e_chain, target_pos, target_rot, q_init
            )

            retry_count += 1

            # 3. 求解失败：直接重试
            if not solve_success:
                if retry_count % 500 == 0:  # 每500次打印一次进度，避免刷屏
                    print(f"   📊 已重试{retry_count}次，仍未求解出有效关节角度...")
                continue

            # 4. 求解成功：检查关节限位
            if not self.is_joint_in_limit(joint_angles):
                continue  # 超出限位，直接舍弃

            # 5. 关节合规：检查位置误差
            error = self.calculate_pos_error(joint_angles, target_pos)
            if error <= self.pos_error_threshold:
                # 双条件满足：判定求解成功
                valid_joint_angles = joint_angles
                final_error = error
                success = True
                print(f"   ✅ 第{retry_count}次求解成功！")
                print(f"   🎯 末端位置误差：{error:.6f} 米（≤ {self.pos_error_threshold:.4f} 米）")
                break
            else:
                # 误差超标，舍弃并重试
                if retry_count % 500 == 0:
                    print(f"   📊 已重试{retry_count}次，当前最小误差：{min(final_error, error):.6f} 米")

        # 重试结束后的处理（无任何裁剪兜底）
        if not success:
            print(f"   ❌ 逆运动学求解失败！已重试{self.max_retry}次")
            print(f"   ❌ 未找到「关节合规 + 误差≤{self.pos_error_threshold:.4f}米」的解")
            print("   建议：")
            print(f"      1. 调整目标位置（当前：x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f}）")
            print("      2. 放宽关节限位范围")
            print(f"      3. 增大误差允许阈值（当前：{self.pos_error_threshold:.4f}米）")
            print("      4. 增加重试次数")
            return

        # 6. 输出最终结果
        print(f"   📏 合规关节角度：{np.round(valid_joint_angles, 4)} 弧度")
        print(f"   🎯 最终末端位置误差：{final_error:.6f} 米")

        # 7. 应用到 MuJoCo（仅合规解）
        self.current_joints = valid_joint_angles
        self.update_joints = True
        print("   ✅ 合规解已更新到 MuJoCo 可视化界面！")

    def stop(self):
        停止可视化
        self.running = False


# =============================================================================================
# 5. 交互式调试入口
# =============================================================================================
def main():
    # --------------------------
    # 配置（修改为你的 MJCF 路径）
    # --------------------------
    MJCF_PATH = "model/universal_robots_ur5e/ur5e.xml"  # 替换为你的XML路径

    # 1. 初始化调试器
    try:
        debugger = UR5eVisualDebugger(MJCF_PATH)
    except Exception as e:
        print(f"❌ 加载 MuJoCo 模型失败：{e}")
        print("请检查 MJCF_PATH 是否正确，或替换为你的 UR5e XML 文件路径")
        return

    # 2. 启动可视化线程
    vis_thread = threading.Thread(target=debugger.start_visualization)
    vis_thread.daemon = True
    vis_thread.start()
    time.sleep(1)  # 等待可视化界面加载

    # 3. 交互式调试菜单
    print("=" * 70)
    print("          UR5e 末端位置调试工具（误差判定 + 限位内猜测值）          ")
    print("=" * 70)
    print("核心规则：")
    print("  ✅ 初始猜测值仅从关节限位范围内选取")
    print(f"  ✅ 求解成功 = 关节合规 + 末端误差≤{debugger.pos_error_threshold:.4f}米")
    print("  ❌ 无角度裁剪，超出条件直接舍弃")
    print("操作说明：")
    print("  1. 输入 1 → 调试关节角度（仅接受合规值）")
    print("  2. 输入 2 → 调试目标位置（误差+限位双判定）")
    print("  3. 输入 3 → 查看当前配置（限位+误差阈值）")
    print("  4. 输入 q → 退出程序")
    print(f"  📝 逆运动学最大重试次数：{debugger.max_retry}")
    print("=" * 70)
    # 初始打印配置
    debugger.print_joint_limits()

    while debugger.running:
        choice = input("\n请输入操作指令（1/2/3/q）：").strip().lower()

        if choice == "1":
            # 调试关节角度
            print("\n请输入6个关节角度（弧度，用空格分隔，示例：0 0 0 0 0 0）：")
            try:
                joint_angles = np.array([float(x) for x in input().split()])
                if len(joint_angles) != 6:
                    print("❌ 必须输入6个关节角度！")
                    continue
                debugger.debug_joint_angles(joint_angles)
            except ValueError:
                print("❌ 输入格式错误！请输入数字，用空格分隔")

        elif choice == "2":
            # 调试目标位置
            print("\n请输入目标末端位置（米，用空格分隔，示例：0.0 -0.5 0.5）：")
            try:
                target_pos = np.array([float(x) for x in input().split()])
                if len(target_pos) != 3:
                    print("❌ 必须输入3个位置坐标（x y z）！")
                    continue
                debugger.debug_target_position(target_pos)
            except ValueError:
                print("❌ 输入格式错误！请输入数字，用空格分隔")

        elif choice == "3":
            # 查看配置
            debugger.print_joint_limits()

        elif choice == "q":
            # 退出程序
            print("\n🔌 正在退出可视化界面...")
            debugger.stop()
            vis_thread.join()
            print("✅ 程序已退出！")
            break

        else:
            print("❌ 无效指令！请输入 1/2/3/q")


if __name__ == "__main__":
    main()
"""

