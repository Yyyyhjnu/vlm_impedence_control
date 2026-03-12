import mujoco.viewer
import ikpy.chain
import transforms3d as tf
import numpy as np
from collections import deque
import matplotlib
# 1. 强制指定支持3D交互式绘图的后端（优先QtAgg，备选TkAgg）
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class ForcePlotter:
    """实时可视化接触力"""

    def __init__(self, update_interval=20):
        plt.ion()  # 开启交互式绘图
        self.fig = plt.figure(figsize=(8, 6))  # 指定窗口大小，避免过小
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_interval = update_interval
        self.frame_count = 0

        # 预初始化绘图元素（放弃quiver复用，改用每次重新绘制，解决兼容性问题）
        self._init_axes()
        # 强制初始绘制+强力刷新，确保初始内容可见
        self._plot_force(np.array([1, 0, 0]))
        plt.show(block=False)  # 非阻塞显示窗口
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _init_axes(self):
        """初始化坐标系（固定不变，无需重复绘制）"""
        self.ax.clear()
        # 固定坐标系配置
        self.ax.scatter(0, 0, 0, color='k', s=20, label='Origin')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X (N)', fontsize=10)
        self.ax.set_ylabel('Y (N)', fontsize=10)
        self.ax.set_zlabel('Z (N)', fontsize=10)
        self.ax.set_title('Real-time Force Direction Visualization', fontsize=12)
        self.ax.view_init(elev=30, azim=45)  # 固定视角，避免抖动
        self.ax.grid(True, alpha=0.3)  # 添加网格，方便观察

    def _plot_force(self, force_vector):
        """重新绘制所有绘图元素（放弃复用，解决3D quiver更新失效问题）"""
        self._init_axes()  # 重置坐标系（清除上一次绘图）
        origin = np.array([0, 0, 0])
        force_magnitude = np.linalg.norm(force_vector)
        force_direction = force_vector / force_magnitude if force_magnitude > 1e-6 else np.array([1, 0, 0])

        # 计算箭头端点
        arrow_tip = force_direction * 1.5  # 力方向箭头长度（固定缩放）
        blue_arrow_tip = arrow_tip + 0.5 * force_direction

        # 1. 绘制主力箭头（红色）
        self.ax.quiver(*origin, *arrow_tip, color='r', arrow_length_ratio=0.1, linewidth=2, label='Main Force')
        # 2. 绘制辅助箭头（蓝色）
        self.ax.quiver(*arrow_tip, *(0.5 * force_direction), color='b', arrow_length_ratio=0.5, linewidth=1)
        # 3. 绘制XY平面投影（绿色虚线）
        self.ax.plot([0, arrow_tip[0]], [0, arrow_tip[1]], [-2, -2], 'g--', alpha=0.7, label='XY Projection')
        # 4. 绘制XZ平面投影（洋红虚线）
        self.ax.plot([0, arrow_tip[0]], [2, 2], [0, arrow_tip[2]], 'm--', alpha=0.7, label='XZ Projection')
        # 5. 绘制力大小指示条（青色实线）
        scaled_force = min(max(force_magnitude / 50, 0), 2)
        self.ax.plot([-2, -2], [2, 2], [0, scaled_force], 'c-', linewidth=3, label='Force Magnitude')
        # 6. 绘制力大小文本（固定位置，避免抖动）
        self.ax.text(-2, 2, scaled_force + 0.1, f'Force: {force_magnitude:.1f} N', color='c', fontsize=9)
        # 显示图例
        self.ax.legend(loc='upper right', fontsize=8)

    def plot_force_vector(self, force_vector):
        """修复刷新逻辑，确保绘图有效显示"""
        if self.frame_count % self.update_interval == 0:
            # 重新绘制力数据
            self._plot_force(force_vector)
            # 强力刷新（解决高频率循环中渲染阻塞问题）
            self.fig.canvas.draw()  # 强制底层渲染
            self.fig.canvas.flush_events()  # 强制窗口刷新
            plt.pause(0.001)  # 轻量阻塞，确保渲染完成（不影响仿真性能）

        # 计数器更新
        self.frame_count += 1
        if self.frame_count >= self.update_interval:
            self.frame_count = 0

    def close(self):
        """释放绘图资源"""
        plt.ioff()
        plt.close(self.fig)
        print("绘图资源已释放")


class ForceController:
    """力控制器：根据力误差调整Z方向位置以保持目标力"""
    
    def __init__(self, target_force=60.0, kp=0.00001, ki=0.0001, kd=0.0005, 
                 max_adjustment=0.005, force_direction='z'):
        """
        初始化力控制器
        
        Args:
            target_force: 目标力大小 (N)
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            max_adjustment: 单次最大位置调整量 (m)
            force_direction: 力控方向 ('z' 表示垂直方向)
        """
        self.target_force = target_force
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_adjustment = max_adjustment
        self.force_direction = force_direction
        
        # PID控制器状态
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_force = 0.0
        
        # 积分限制（防止积分饱和）
        self.integral_limit = 0.1
        
    def update(self, current_force):
        """
        根据当前力计算位置调整量
        
        Args:
            current_force: 当前力向量 [Fx, Fy, Fz] 
        
        Returns:
            position_adjustment: Z方向的位置调整量 (m)
        """
      
        force_magnitude = np.linalg.norm(np.array(current_force))
        
        # 计算力误差（目标力 - 当前力）
        # 如果误差为正，说明当前力小于目标力，需要增加接触力（向下移动，减小Z坐标）
        # 如果误差为负，说明当前力大于目标力，需要减少接触力（向上移动，增加Z坐标）
        error = self.target_force - force_magnitude
        
        # PID控制
        # 比例项
        p_term = self.kp * error
        
        # 积分项（带限制）
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral_error
        
        # 微分项
        d_error = error - self.last_error
        d_term = self.kd * d_error
        
        # 总调整量计算
        # 根据误差符号判断移动方向，PID输出只决定调整幅度
        # 如果误差为正（当前力 < 目标力），需要增加接触力，应该向下移动（减小Z坐标）
        # 如果误差为负（当前力 > 目标力），需要减少接触力，应该向上移动（增加Z坐标）
        # 注意：在MuJoCo中，Z坐标增加表示向上移动，Z坐标减小表示向下移动
        
        # 计算PID输出（幅度）
        pid_output = p_term + i_term + d_term
        
        # 根据误差符号确定调整方向
        # error > 0: 需要增加力 → 向下移动 → 调整量为负
        # error < 0: 需要减少力 → 向上移动 → 调整量为正
        # error = 0: 无需调整
        if abs(error) < 1e-6:  # 误差接近0，不调整
            adjustment = 0.0
        else:
            # 使用误差符号确定方向，PID输出决定幅度
            adjustment = -np.sign(error) * abs(pid_output)
        
        # 限制调整量
        adjustment = np.clip(adjustment, -self.max_adjustment, self.max_adjustment)
        
        
        # 更新状态
        self.last_error = error
        self.last_force = force_magnitude
        
        return adjustment
    
    def reset(self):
        """重置控制器状态"""
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_force = 0.0


class ForceSensor:
    def __init__(self, model, data, window_size=100):
        self.model = model
        self.data = data
        self.window_size = window_size
        self.force_history = deque(maxlen=window_size)
        # 保存所有原始力数据用于最终绘图
        self.all_force_data = []

    def filter(self):
        """获取并滑动平均滤波力传感器数据(传感器坐标系下)"""
        force_local_raw = self.data.sensordata[:3].copy() * -1
        self.force_history.append(force_local_raw)
        # 保存所有原始数据
        self.all_force_data.append(force_local_raw.copy())
        filtered_force = np.mean(self.force_history, axis=0)
        return filtered_force

    def get_all_data(self):
        """获取所有记录的力数据"""
        return np.array(self.all_force_data)


def plot_force_history(force_data, save_path=None):
    """
    生成传感器数据的平面图
    
    Args:
        force_data: 力数据数组，形状为 (n_samples, 3)，包含 [Fx, Fy, Fz]
        save_path: 可选，保存图片的路径
    """
    if len(force_data) == 0:
        print("警告: 没有力数据可绘制")
        return
    
    # 转换为numpy数组
    force_data = np.array(force_data)
    n_samples = len(force_data)
    
    # 创建时间轴（假设每个样本对应一个仿真步）
    time_axis = np.arange(n_samples)
    
    # 计算合力大小
    force_magnitude = np.linalg.norm(force_data, axis=1)
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('传感器力数据记录', fontsize=14, fontweight='bold')
    
    # 子图1: Fx随时间变化
    axes[0, 0].plot(time_axis, force_data[:, 0], 'r-', linewidth=1.5, label='Fx')
    axes[0, 0].set_xlabel('时间步', fontsize=10)
    axes[0, 0].set_ylabel('力 (N)', fontsize=10)
    axes[0, 0].set_title('X方向力 (Fx)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: Fy随时间变化
    axes[0, 1].plot(time_axis, force_data[:, 1], 'g-', linewidth=1.5, label='Fy')
    axes[0, 1].set_xlabel('时间步', fontsize=10)
    axes[0, 1].set_ylabel('力 (N)', fontsize=10)
    axes[0, 1].set_title('Y方向力 (Fy)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: Fz随时间变化
    axes[1, 0].plot(time_axis, force_data[:, 2], 'b-', linewidth=1.5, label='Fz')
    axes[1, 0].set_xlabel('时间步', fontsize=10)
    axes[1, 0].set_ylabel('力 (N)', fontsize=10)
    axes[1, 0].set_title('Z方向力 (Fz)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 合力大小随时间变化
    axes[1, 1].plot(time_axis, force_magnitude, 'm-', linewidth=1.5, label='合力大小')
    axes[1, 1].set_xlabel('时间步', fontsize=10)
    axes[1, 1].set_ylabel('力 (N)', fontsize=10)
    axes[1, 1].set_title('合力大小 |F|', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"力数据图表已保存到: {save_path}")
    
    # 显示图表
    plt.show(block=False)
    print(f"已生成力数据记录图，共 {n_samples} 个数据点")
    print(f"力数据统计:")
    print(f"  Fx: 均值={np.mean(force_data[:, 0]):.3f} N, 最大值={np.max(force_data[:, 0]):.3f} N, 最小值={np.min(force_data[:, 0]):.3f} N")
    print(f"  Fy: 均值={np.mean(force_data[:, 1]):.3f} N, 最大值={np.max(force_data[:, 1]):.3f} N, 最小值={np.min(force_data[:, 1]):.3f} N")
    print(f"  Fz: 均值={np.mean(force_data[:, 2]):.3f} N, 最大值={np.max(force_data[:, 2]):.3f} N, 最小值={np.min(force_data[:, 2]):.3f} N")
    print(f"  合力大小: 均值={np.mean(force_magnitude):.3f} N, 最大值={np.max(force_magnitude):.3f} N, 最小值={np.min(force_magnitude):.3f} N")


def inverse_kinematics_with_limits(chain, target_pos, target_orientation, initial_position,
                                   joint_lower_limits=None, joint_upper_limits=None,
                                   max_iterations=10, position_tolerance=1e-3,verbose=False):
    """
    带关节角度限制的逆运动学求解函数

    Args:
        chain: ikpy链对象
        target_pos: 目标位置 [x, y, z]
        target_orientation: 目标姿态（旋转矩阵或欧拉角）
        initial_position: 初始关节角度猜测值（完整链，包括非活动关节）
        joint_lower_limits: 关节角度下界（仅活动关节，长度为6）
        joint_upper_limits: 关节角度上界（仅活动关节，长度为6）
        max_iterations: 最大迭代次数（如果裁剪后误差太大，尝试重新求解）
        position_tolerance: 位置误差容忍度（米）\
        verbose:是否打印详细信息

    Returns:
        joint_angles: 满足关节限制的关节角度（完整链）
        success: 是否成功找到满足限制的解
        position_error: 实际位置与目标位置的误差
    """
    # 默认关节限制（UR5e的标准限制：-π 到 π）
    if joint_lower_limits is None:
        joint_lower_limits = np.array([-np.pi] * 6)
    if joint_upper_limits is None:
        joint_upper_limits = np.array([np.pi] * 6)

    joint_lower_limits = np.array(joint_lower_limits)
    joint_upper_limits = np.array(joint_upper_limits)

    best_solution = None
    best_error = float('inf')
    best_position_error = float('inf')

    # 尝试多次求解，每次使用不同的初始猜测
    for iteration in range(max_iterations):
        # 使用逆运动学求解
        try:
            joint_angles = chain.inverse_kinematics(
                target_pos,
                target_orientation,
                "all",
                initial_position=initial_position
            )
        except Exception as e:
            if verbose:
                print(f"迭代 {iteration + 1}: 逆运动学求解失败: {e}")
            continue

        # 提取活动关节角度（索引2到7，共6个）
        active_joints = joint_angles[2:-1]

        # 检查并裁剪到关节限制范围内
        clamped_joints = np.clip(active_joints, joint_lower_limits, joint_upper_limits)

        # 检查是否有关节被裁剪
        was_clamped = not np.allclose(active_joints, clamped_joints, atol=1e-6)

        # 重新构造完整关节角度向量
        clamped_full_angles = np.concatenate([
            joint_angles[:2],  # 前两个非活动关节
            clamped_joints,  # 裁剪后的活动关节
            joint_angles[-1:]  # 最后一个非活动关节
        ])

        # 通过正运动学验证裁剪后的位置
        fk_result = chain.forward_kinematics(clamped_full_angles)
        actual_pos = fk_result[:3, 3]
        position_error = np.linalg.norm(np.array(target_pos) - actual_pos)

        # 计算与初始猜测的偏差（用于选择最接近初始值的解）
        initial_active = initial_position[2:-1]
        deviation = np.linalg.norm(clamped_joints - initial_active)

        if verbose:
            print(f"迭代 {iteration + 1}:")
            print(f"  原始关节角度: {np.round(active_joints, 3)}")
            if was_clamped:
                print(f"  裁剪后角度: {np.round(clamped_joints, 3)}")
                clamped_indices = np.where(np.abs(active_joints - clamped_joints) > 1e-6)[0]
                print(f"  被裁剪的关节索引: {clamped_indices}")
            print(f"  位置误差: {position_error:.6f} 米")
            print(f"  与初始猜测偏差: {deviation:.4f} 弧度")

        # 如果位置误差在容忍范围内，优先选择偏差最小的解
        if position_error < position_tolerance:
            if deviation < best_error:
                best_solution = clamped_full_angles.copy()
                best_error = deviation
                best_position_error = position_error
        elif best_solution is None or position_error < best_position_error:
            # 如果所有解都不满足精度要求，选择位置误差最小的
            best_solution = clamped_full_angles.copy()
            best_position_error = position_error
            best_error = deviation

        # 如果找到满足精度要求的解，可以提前退出
        if position_error < position_tolerance and not was_clamped:
            if verbose:
                print(f"找到满足精度要求的解（无需裁剪）")
            break

        # 为下一次迭代生成新的初始猜测（在限制范围内随机扰动）
        if iteration < max_iterations - 1:
            perturbation = np.random.uniform(-0.1, 0.1, size=6)
            perturbed_active = np.clip(
                initial_active + perturbation,
                joint_lower_limits,
                joint_upper_limits
            )
            initial_position = np.concatenate([
                initial_position[:2],
                perturbed_active,
                initial_position[-1:]
            ])

    if best_solution is None:
        if verbose:
            print("警告: 未能找到满足关节限制的解")
        return initial_position, False, float('inf')

    success = best_position_error < position_tolerance
    if verbose:
        print(f"\n最终结果:")
        print(f"  成功: {success}")
        print(f"  位置误差: {best_position_error:.6f} 米")
        print(f"  最终关节角度: {np.round(best_solution[2:-1], 3)}")

    return best_solution, success, best_position_error


class CartesianSpaceTrajectory:
    """笛卡尔坐标空间下的线性插值轨迹（仅沿y轴运动，支持Z方向力控调整）"""

    def __init__(self, start_pos, end_pos, start_orientation, chain, ref_pos, steps, ref_angle,
                 joint_lower_limits=None, joint_upper_limits=None, use_joint_limits=True,
                 force_controller=None):
        """
        初始化笛卡尔空间轨迹
        Args:
            start_pos: 起始位置 [x, y, z]
            end_pos: 目标位置 [x, y, z]
            start_orientation: 起始姿态（欧拉角 [rx, ry, rz] 或旋转矩阵）
            chain: 逆运动学链对象
            ref_pos: 逆运动学参考位置
            steps: 插值步数
            ref_angle: 参考关节角度（完整链）
            joint_lower_limits: 关节角度下界（仅活动关节，长度为6）
            joint_upper_limits: 关节角度上界（仅活动关节，长度为6）
            use_joint_limits: 是否使用关节限制
            force_controller: 力控制器对象（可选）
        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        # 只沿y轴运动，x和z使用起始位置的坐标（保持不变）
        self.fixed_x = self.start_pos[0]
        self.base_z = self.start_pos[2]  # 基础Z坐标
        self.current_z = self.base_z  # 当前Z坐标（会被力控调整）

        self.start_orientation = start_orientation
        self.chain = chain
        self.ref_pos = ref_pos
        self.steps = steps
        # 只计算y方向的步长
        self.step_y = (self.end_pos[1] - self.start_pos[1]) / self.steps
        self.current_step = 0  # 当前轨迹步数
        self.current_waypoint_joints = None
        self.current_cartesian_pos = np.array([self.fixed_x, self.start_pos[1], self.current_z])
        self.ref_angle = ref_angle

        # 关节限制参数
        self.use_joint_limits = use_joint_limits
        self.joint_lower_limits = joint_lower_limits
        self.joint_upper_limits = joint_upper_limits
        
        # 力控制器
        self.force_controller = force_controller
        # 用于调试的调整量记录
        self.last_z_adjustment = 0.0
        self.calculated_z_adjustment = 0.0

    def _get_trajectory_point(self, step_index):
        """获取指定步数的轨迹点（仅y轴变化，Z坐标由力控动态调整）"""
        # 只改变y坐标，x保持不变，z使用当前力控调整后的值
        y_pos = self.start_pos[1] + self.step_y * step_index
        cartesian_pos = np.array([self.fixed_x, y_pos, self.current_z])
        return cartesian_pos
    
    def update_z_position(self, z_adjustment):
        """
        根据力控调整更新Z坐标
        
        Args:
            z_adjustment: Z方向的位置调整量 (m)
        """
        old_z = self.current_z
        self.current_z += z_adjustment
        # 限制Z坐标范围（防止过度调整，增大范围以允许更大的调整）
        self.current_z = np.clip(self.current_z, self.base_z - 0.15, self.base_z + 0.15)
        # 返回实际调整量（考虑clip后的限制）
        actual_adjustment = self.current_z - old_z
        return actual_adjustment

    def get_next_waypoint(self, qpos, current_force=None):
        """
        获取下一个关节空间路径点（带关节限制和力控）
        Args:
            qpos: 当前关节角度
            current_force: 当前力传感器读数（可选，用于力控）
        Returns:
            下一个目标关节角度
        """
        # 如果提供了力传感器数据且有力控制器，更新Z坐标
        if current_force is not None and self.force_controller is not None:
            z_adjustment = self.force_controller.update(current_force)
            # 存储计算出的调整量（应用前）
            self.calculated_z_adjustment = z_adjustment
            actual_adjustment = self.update_z_position(z_adjustment)
            # 更新当前笛卡尔位置（Z坐标已调整）
            self.current_cartesian_pos[2] = self.current_z
            # 存储实际调整量用于调试
            self.last_z_adjustment = actual_adjustment
        
        # 检查是否到达当前路径点（使用关节空间距离）
        if self.current_waypoint_joints is not None:
            if np.allclose(qpos, self.current_waypoint_joints, atol=0.02):
                # 移动到下一个轨迹点
                if self.current_step < self.steps:
                    self.current_step += 1
                    # 获取下一个笛卡尔坐标点（Y方向，Z坐标使用力控调整后的值）
                    next_cartesian_pos = self._get_trajectory_point(self.current_step)
                    self.current_cartesian_pos = next_cartesian_pos.copy()
                    # 使用逆运动学计算关节角度
                    orientation = self.start_orientation
                    if isinstance(orientation, list) or isinstance(orientation, tuple):
                        orientation = tf.euler.euler2mat(*orientation)

                    # 根据是否使用关节限制选择不同的求解方法
                    if self.use_joint_limits and self.joint_lower_limits is not None:
                        joint_angles, success, pos_error = inverse_kinematics_with_limits(
                            chain=self.chain,
                            target_pos=next_cartesian_pos,
                            target_orientation=orientation,
                            initial_position=self.ref_angle,
                            joint_lower_limits=self.joint_lower_limits,
                            joint_upper_limits=self.joint_upper_limits,
                            max_iterations=3,  # 轨迹点计算时减少迭代次数以提高速度
                            position_tolerance=1e-3,
                            verbose=False
                        )
                    else:
                        joint_angles = self.chain.inverse_kinematics(
                            next_cartesian_pos,
                            orientation,
                            "all",
                            initial_position=self.ref_angle
                        )

                    self.ref_angle = joint_angles[0:9]
                    self.current_waypoint_joints = joint_angles[2:-1]
            else:
                # 即使未到达路径点，如果有力控，也需要根据当前力更新Z坐标并重新计算逆运动学
                # 更新当前轨迹点的Z坐标（保持Y坐标不变）
                if current_force is not None and self.force_controller is not None:
                    # 更新当前笛卡尔位置的Z坐标
                    self.current_cartesian_pos[2] = self.current_z
                    # 使用当前笛卡尔位置（Z坐标已更新）重新计算逆运动学
                    orientation = self.start_orientation
                    if isinstance(orientation, list) or isinstance(orientation, tuple):
                        orientation = tf.euler.euler2mat(*orientation)
                    
                    if self.use_joint_limits and self.joint_lower_limits is not None:
                        joint_angles, success, pos_error = inverse_kinematics_with_limits(
                            chain=self.chain,
                            target_pos=self.current_cartesian_pos,
                            target_orientation=orientation,
                            initial_position=self.ref_angle,
                            joint_lower_limits=self.joint_lower_limits,
                            joint_upper_limits=self.joint_upper_limits,
                            max_iterations=3,
                            position_tolerance=1e-3,
                            verbose=False
                        )
                    else:
                        joint_angles = self.chain.inverse_kinematics(
                            self.current_cartesian_pos,
                            orientation,
                            "all",
                            initial_position=self.ref_angle
                        )
                    self.ref_angle = joint_angles[0:9]
                    self.current_waypoint_joints = joint_angles[2:-1]
        else:
            # 初始化：计算起始位置的关节角度
            orientation = self.start_orientation
            if isinstance(orientation, list) or isinstance(orientation, tuple):
                orientation = tf.euler.euler2mat(*orientation)

            # 根据是否使用关节限制选择不同的求解方法
            if self.use_joint_limits and self.joint_lower_limits is not None:
                joint_angles, success, pos_error = inverse_kinematics_with_limits(
                    chain=self.chain,
                    target_pos=self.start_pos,
                    target_orientation=orientation,
                    initial_position=self.ref_pos,
                    joint_lower_limits=self.joint_lower_limits,
                    joint_upper_limits=self.joint_upper_limits,
                    max_iterations=3,
                    position_tolerance=1e-3,
                    verbose=False
                )
            else:
                joint_angles = self.chain.inverse_kinematics(
                    self.start_pos,
                    orientation,
                    "all",
                    initial_position=self.ref_pos
                )
            self.current_waypoint_joints = joint_angles[2:-1]

        return self.current_waypoint_joints


def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene-30.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e_orig.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])

    joint_lower_limits = np.array([-np.pi, -np.pi, -np.pi, -2.5, -3.2, -np.pi])
    joint_upper_limits = np.array([np.pi, np.pi, np.pi, -0.8, -3.1, np.pi])

    #初始点计算

    start_pos = [0.0, 0.3, 0.125]
    start_euler = [0, 0, 0]
    start_ref_pos = [0, 0, -1.63, -1.37, 2.47, -1.54, -3.14, 0, 0]
    start_orientation = tf.euler.euler2mat(*start_euler)

    print("计算初始位置的逆运动学解（带关节限制）...")
    start_joints_angle, success, pos_error = inverse_kinematics_with_limits(
        chain=my_chain,
        target_pos=start_pos,
        target_orientation=start_orientation,
        initial_position=start_ref_pos,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        max_iterations=5,
        position_tolerance=1e-3,
        verbose=True
    )
    start_joints=start_joints_angle[2:-1]

    print(f"最终设置的关节角度: {np.round(data.qpos[:6], 3)}")

    #终点
    ee_pos = [0.0, 0.5, 0.125]  # 先设置在平面上方，与起始位置一致
    ee_euler = [0, 0, 0]
    ee_ref_pos = [0, 0, -1.63, -1.04, 1.9, -1.21, -3.14, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    print("\n计算目标位置的逆运动学解（带关节限制）...")
    end_joints_angles, success, pos_error = inverse_kinematics_with_limits(
        chain=my_chain,
        target_pos=ee_pos,
        target_orientation=ee_orientation,
        initial_position=ee_ref_pos,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        max_iterations=10,
        position_tolerance=1e-3,
        verbose=True
    )
    end_joints = end_joints_angles[2:-1]

    # 避障点
    barrier_avoidance_point = [0, 0.2, 1]
    bap_euler = [0, 0, 0]
    bap_ref_pos = [0, 0, -1.63, -1.76, 0.822, -1.76, -3.14, 0, 0]
    bap_orientation = tf.euler.euler2mat(*bap_euler)

    print("\n计算目标位置的逆运动学解（带关节限制）...")
    barrier_avoidance_joints_angles, success, pos_error = inverse_kinematics_with_limits(
        chain=my_chain,
        target_pos=barrier_avoidance_point,
        target_orientation=bap_orientation,
        initial_position=bap_ref_pos,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        max_iterations=10,
        position_tolerance=1e-3,
        verbose=True
    )
    bap_joints = barrier_avoidance_joints_angles[2:-1]


    # 初始化力控制器
    force_controller = ForceController(
        target_force=60.0,  # 目标力60N
        kp=0.000005,          # 比例增益（可根据实际情况调整）
        ki=0.000005,         # 积分增益
        kd=0.00005,          # 微分增益
        max_adjustment=0.003  # 单次最大调整量3mm
    )
    
    # 初始化轨迹、力传感器、绘图器
    # 使用笛卡尔坐标空间轨迹（仅沿y轴运动，带关节限制和力控）
    cartesian_trajectory = CartesianSpaceTrajectory(
        start_pos=start_pos,
        end_pos=ee_pos,
        start_orientation=start_orientation,
        chain=my_chain,
        ref_pos=start_ref_pos,
        steps=100,
        ref_angle=start_joints_angle[0:9],
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        use_joint_limits=True,  # 启用关节限制
        force_controller=force_controller  # 传入力控制器
    )
    force_sensor = ForceSensor(model, data)
    force_plotter = ForcePlotter(update_interval=20)

    print("\n" + "="*60)
    print("力控模式已启用")
    print(f"目标力: {force_controller.target_force} N")
    print(f"控制参数: Kp={force_controller.kp}, Ki={force_controller.ki}, Kd={force_controller.kd}")
    print("="*60 + "\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer_init(viewer)
            # 初始化：设置初始关节角度
            data.qpos[:6] = start_joints
            mujoco.mj_forward(model, data)
            
            # 用于控制打印频率的计数器
            print_counter = 0
            
            while viewer.is_running():
                # 更新正向运动学以获取当前传感器数据
                mujoco.mj_forward(model, data)
                
                # 获取滤波后的力数据
                filtered_force = force_sensor.filter()
                
                # 更新关节目标值（从笛卡尔空间轨迹获取，传入力传感器数据用于力控）
                waypoint = cartesian_trajectory.get_next_waypoint(data.qpos[:6], current_force=filtered_force)
                
                # 设置控制输入（位置控制）
                data.ctrl[:6] = waypoint
                
                # 打印调试信息（降低打印频率）
                print_counter += 1
                if print_counter % 50 == 0:  # 每50步打印一次
                    force_z = filtered_force[2] if len(filtered_force) >= 3 else 0.0
                    force_error = force_controller.target_force - force_z
                    current_z = cartesian_trajectory.current_z
                    calculated_adjustment = getattr(cartesian_trajectory, 'calculated_z_adjustment', 0.0)
                    actual_adjustment = getattr(cartesian_trajectory, 'last_z_adjustment', 0.0)
                    # 计算PID各项用于调试
                    p_term = force_controller.kp * force_error
                    i_term = force_controller.ki * force_controller.integral_error
                    d_term = force_controller.kd * (force_error - force_controller.last_error)
                    # 检查调整方向
                    direction = "向下" if calculated_adjustment < 0 else "向上" if calculated_adjustment > 0 else "无"
                    print(f"力控状态 - Z坐标: {current_z:.4f}m (基础: {cartesian_trajectory.base_z:.4f}m, 范围: [{cartesian_trajectory.base_z-0.15:.4f}, {cartesian_trajectory.base_z+0.15:.4f}]m), ")
                    print(f"          Z方向力: {force_z:.2f}N, 目标力: {force_controller.target_force}N, 力误差: {force_error:.2f}N")
                    print(f"          计算调整量: {calculated_adjustment:.6f}m ({direction}), 实际调整量: {actual_adjustment:.6f}m")
                    print(f"          PID项: P={p_term:.6f}, I={i_term:.6f}, D={d_term:.6f}, 总和={p_term+i_term+d_term:.6f}")
                
                # 绘制力向量
                force_plotter.plot_force_vector(filtered_force)
                
                # 运行MuJoCo仿真步
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        force_plotter.close()
        # 程序结束后生成传感器数据的平面图
        print("\n正在生成传感器数据记录图...")
        all_force_data = force_sensor.get_all_data()
        plot_force_history(all_force_data, save_path='force_sensor_data.png')

if __name__ == "__main__":
    main()