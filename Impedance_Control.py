import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time


def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


def plot_results(q_history, qdot_history, torque_history, external_force_history, eef_position_history, total_time, dt, num_steps):
    """绘制仿真结果"""
    time_axis = np.arange(0, total_time, dt)[:num_steps]
    
    plt.figure(figsize=(16, 12))
    
    # 绘制关节角度
    plt.subplot(5, 1, 1)
    for j in range(q_history.shape[1]):
        plt.plot(time_axis, q_history[:num_steps, j], label=f'Joint {j+1}')
    plt.title('Joint Angles')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制关节速度
    plt.subplot(5, 1, 2)
    for j in range(qdot_history.shape[1]):
        plt.plot(time_axis, qdot_history[:num_steps, j], label=f'Joint {j+1}')
    plt.title('Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制控制扭矩
    plt.subplot(5, 1, 3)
    for j in range(torque_history.shape[1]):
        plt.plot(time_axis, torque_history[:num_steps, j], label=f'Joint {j+1}')
    plt.title('Control Torques')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N.m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制外部力
    plt.subplot(5, 1, 4)
    for j in range(external_force_history.shape[1]):
        plt.plot(time_axis, external_force_history[:num_steps, j], label=f'External Force Joint {j+1}')
    plt.title('External Forces Applied to Joints')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N.m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制末端位置
    plt.subplot(5, 1, 5)
    if eef_position_history.shape[1] >= 3:
        plt.plot(time_axis, eef_position_history[:num_steps, 0], label='X', linewidth=2)
        plt.plot(time_axis, eef_position_history[:num_steps, 1], label='Y', linewidth=2)
        plt.plot(time_axis, eef_position_history[:num_steps, 2], label='Z', linewidth=2)
        plt.title('End-Effector Position (Should Remain Constant)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


class MouseDragForceController:
    """鼠标拖动施加外力控制器"""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.num_joints = model.nu
        
        # 鼠标拖动状态
        self.is_dragging = False
        self.drag_joint = None  # 当前拖动的关节索引
        self.last_mouse_y = None  # 上次鼠标Y位置
        self.drag_start_y = None  # 拖动起始Y位置
        
        # 外力参数
        self.force_scale = 100.0  # 拖动距离到外力的缩放因子
        self.max_force = 150.0  # 最大外力 (N.m)
        
        # 当前施加的外力
        self.current_forces = np.zeros(model.nv)
        
        # 打印使用说明
        self._print_instructions()
    
    def _print_instructions(self):
        """打印操作说明"""
        print("\n" + "="*60)
        print("鼠标拖动外力控制说明:")
        print("="*60)
        print("操作方式:")
        print("  1. 按住鼠标左键点击并拖动关节来施加外力")
        print("  2. 拖动方向决定外力方向（向上为正，向下为负）")
        print("  3. 拖动距离决定外力大小（拖动越远，外力越大）")
        print("  4. 松开鼠标左键停止施加外力")
        print("\n提示:")
        print("  - 点击关节附近的区域来选择关节")
        print("  - 拖动速度越快，外力越大")
        print("="*60 + "\n")
    
    def update_from_mouse(self, mouse_button, mouse_x, mouse_y, viewport_height):
        """根据鼠标状态更新外力"""
        # 检测鼠标左键按下
        if mouse_button == 0:  # 左键按下
            if not self.is_dragging:
                # 开始拖动，选择关节
                self._start_drag(mouse_y, viewport_height)
            else:
                # 继续拖动，更新外力
                self._update_drag(mouse_y, viewport_height)
        else:
            # 鼠标释放
            if self.is_dragging:
                self._stop_drag()
    
    def _start_drag(self, mouse_y, viewport_height):
        """开始拖动"""
        # 根据鼠标Y坐标选择关节（简化方法）
        # 屏幕从上到下对应关节0-5
        screen_y_ratio = mouse_y / viewport_height  # 0-1
        joint_idx = int(screen_y_ratio * self.num_joints)
        joint_idx = np.clip(joint_idx, 0, self.num_joints - 1)
        
        self.is_dragging = True
        self.drag_joint = joint_idx
        self.drag_start_y = mouse_y
        self.last_mouse_y = mouse_y
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
        print(f"开始拖动关节 {self.drag_joint+1} ({joint_names[self.drag_joint]})")
    
    def _update_drag(self, mouse_y, viewport_height):
        """更新拖动外力"""
        if self.drag_joint is None:
            return
        
        # 计算拖动距离（相对于起始位置）
        dy = mouse_y - self.drag_start_y
        
        # 转换为归一化距离（-1到1）
        normalized_dy = -dy / (viewport_height / 2.0)  # 负号是因为屏幕Y轴向下
        
        # 计算外力大小
        force_magnitude = normalized_dy * self.force_scale
        
        # 限制外力大小
        force_magnitude = np.clip(force_magnitude, -self.max_force, self.max_force)
        
        # 应用外力到选定的关节
        if 0 <= self.drag_joint < self.model.nv:
            self.current_forces[self.drag_joint] = force_magnitude
        
        self.last_mouse_y = mouse_y
    
    def _stop_drag(self):
        """停止拖动"""
        if self.is_dragging and self.drag_joint is not None:
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
            print(f"停止拖动关节 {self.drag_joint+1} ({joint_names[self.drag_joint]})")
            # 停止对该关节施加外力
            self.current_forces[self.drag_joint] = 0.0
        
        self.is_dragging = False
        self.drag_joint = None
        self.last_mouse_y = None
        self.drag_start_y = None
    
    def get_forces(self):
        """获取当前应该施加的外力数组"""
        return self.current_forces.copy()
    
    def reset(self):
        """重置所有外力"""
        self.current_forces.fill(0.0)
        self._stop_drag()


def compute_null_space_projection(J, lambda_reg=1e-6):
    """
    计算零空间投影矩阵
    
    Args:
        J: 雅可比矩阵 (m x n)，m是任务空间维度，n是关节空间维度
        lambda_reg: 正则化参数，避免奇异
    
    Returns:
        N: 零空间投影矩阵 (n x n)
    """
    n = J.shape[1]
    I = np.eye(n)
    
    # 计算 (J * J^T + lambda*I)^(-1)
    JJT = J @ J.T
    JJT_reg = JJT + lambda_reg * np.eye(JJT.shape[0])
    
    # 计算零空间投影矩阵: N = I - J^T * (J * J^T + lambda*I)^(-1) * J
    try:
        JJT_inv = np.linalg.inv(JJT_reg)
        N = I - J.T @ JJT_inv @ J
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        JJT_pinv = np.linalg.pinv(JJT_reg)
        N = I - J.T @ JJT_pinv @ J
    
    return N


def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene-30.xml')
    data = mujoco.MjData(model)
    
    # 获取末端执行器site的ID
    eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
    if eef_site_id < 0:
        raise ValueError("找不到末端执行器site: attachment_site")
    
    # 目标关节角度（用于零空间控制）
    q_desired = np.array([-1.63, -1.37, 2.47, 0.662, -0.88, 0.66])
    
    # 设置初始关节角度
    data.qpos[:model.nu] = q_desired
    data.qvel[:model.nu] = np.zeros(model.nu)
    
    # 初始化正向运动学，获取目标末端位置
    mujoco.mj_forward(model, data)
    x_desired = data.site(eef_site_id).xpos.copy()  # 目标末端位置
    print(f"目标末端位置: [{x_desired[0]:.4f}, {x_desired[1]:.4f}, {x_desired[2]:.4f}]")
    
    # 零空间阻抗控制参数
    # 任务空间（末端位置）参数 - 高刚度保持位置
    Kp_task = 5000.0  # 任务空间位置刚度（高）
    Kd_task = 100.0   # 任务空间速度阻尼（高）
    
    # 零空间（关节角度）参数 - 低刚度允许变化
    Kp_null = 5.0     # 零空间位置刚度（低，允许关节角度变化）
    Kd_null = 2.0     # 零空间速度阻尼（低）
    
    print(f"\n零空间阻抗控制参数:")
    print(f"  任务空间刚度 Kp_task: {Kp_task}")
    print(f"  任务空间阻尼 Kd_task: {Kd_task}")
    print(f"  零空间刚度 Kp_null: {Kp_null}")
    print(f"  零空间阻尼 Kd_null: {Kd_null}")
    print(f"\n说明: 外力作用时，关节角度可以变化，但末端位置将保持稳定\n")
    
    # ========== 鼠标拖动外力控制器 ==========
    force_controller = MouseDragForceController(model, data)
    # ========================================
    
    # 仿真参数
    total_time = 30  # 总仿真时间（秒）
    dt = model.opt.timestep  # 仿真时间步长
    num_steps = int(total_time / dt)
    current_time = 0.0  # 当前仿真时间
    
    # 存储数据
    q_history = np.zeros((num_steps, model.nu))
    qdot_history = np.zeros((num_steps, model.nu))
    torque_history = np.zeros((num_steps, model.nu))
    external_force_history = np.zeros((num_steps, model.nv))  # nv是速度空间维度
    eef_position_history = np.zeros((num_steps, 3))  # 存储末端位置
    index = 0
    
    # 启用数据记录
    enable_data_recording = True
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer_init(viewer)
            
            # 存储鼠标状态
            last_mouse_button = None
            
            while viewer.is_running():
                # 尝试从viewer获取鼠标状态
                # 注意：MuJoCo的viewer可能不直接暴露鼠标状态
                # 这里我们使用一个变通方法：通过viewer的内部属性
                mouse_button = None
                mouse_x = 0
                mouse_y = 0
                
                # 尝试获取鼠标状态（如果viewer支持）
                try:
                    # 检查viewer是否有鼠标相关的属性
                    # 不同版本的MuJoCo可能有不同的属性名
                    if hasattr(viewer, 'mouse_button'):
                        mouse_button = viewer.mouse_button
                        mouse_x = getattr(viewer, 'mouse_x', 0)
                        mouse_y = getattr(viewer, 'mouse_y', 0)
                    elif hasattr(viewer, '_mouse_button'):
                        mouse_button = viewer._mouse_button
                        mouse_x = getattr(viewer, '_mouse_x', 0)
                        mouse_y = getattr(viewer, '_mouse_y', 0)
                except:
                    pass
                
                # 如果检测到鼠标状态变化，更新外力控制器
                if mouse_button is not None:
                    force_controller.update_from_mouse(
                        mouse_button, mouse_x, mouse_y, viewer.viewport.height
                    )
                    last_mouse_button = mouse_button
                elif last_mouse_button is not None:
                    # 如果之前有鼠标按下，但现在检测不到，说明已释放
                    force_controller.update_from_mouse(-1, 0, 0, viewer.viewport.height)
                    last_mouse_button = None
                
                # 更新正向运动学
                mujoco.mj_forward(model, data)
                
                # 读取当前关节角度和速度
                q = data.qpos[:model.nu]
                qdot = data.qvel[:model.nu]
                
                # 获取当前末端位置和速度
                x_current = data.site(eef_site_id).xpos.copy()
                
                # 计算任务空间雅可比矩阵（位置部分，3 x nv）
                jac_pos = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jac_pos, None, eef_site_id)
                J = jac_pos[:, :model.nu]  # 只取前nu列（驱动关节）
                
                # 计算任务空间误差和速度
                x_error = x_desired - x_current
                xdot = J @ qdot  # 任务空间速度
                
                # 任务空间控制力（在任务空间中）
                F_task = Kp_task * x_error - Kd_task * xdot
                
                # 将任务空间控制力转换到关节空间
                tau_task = J.T @ F_task
                
                # 计算零空间投影矩阵
                N = compute_null_space_projection(J)
                
                # 零空间控制（允许关节角度变化，但不影响末端位置）
                q_error = q_desired - q
                tau_null = N @ (Kp_null * q_error - Kd_null * qdot)
                
                # 总控制扭矩 = 任务空间控制 + 零空间控制
                torque = tau_task + tau_null
                
                # 设置控制输入
                data.ctrl[:] = torque
                
                # ========== 应用鼠标拖动外部力到关节 ==========
                # 从鼠标拖动控制器获取当前外力
                external_force = force_controller.get_forces()
                
                # 应用外部力到MuJoCo数据
                data.qfrc_applied[:] = external_force
                # ==============================================
                
                # 记录数据
                if enable_data_recording and index < num_steps:
                    q_history[index] = q
                    qdot_history[index] = qdot
                    torque_history[index] = torque
                    external_force_history[index] = external_force
                    eef_position_history[index] = x_current.copy()
                    index += 1
                
                # 运行MuJoCo仿真步
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 更新仿真时间
                current_time += dt
                
                # 可选：打印调试信息
                if index % 500 == 0:  # 减少打印频率
                    active_forces = [f"J{i+1}:{f:.1f}" for i, f in enumerate(external_force[:model.nu]) if abs(f) > 0.01]
                    force_str = ", ".join(active_forces) if active_forces else "无"
                    pos_error = np.linalg.norm(x_error)
                    print(f"时间: {current_time:.2f}s | 激活的外力: {force_str} | 末端位置误差: {pos_error:.4f}m")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 绘制结果
        if enable_data_recording and index > 0:
            print("\n正在生成仿真结果图表...")
            plot_results(q_history, qdot_history, torque_history, external_force_history, eef_position_history, total_time, dt, index)
            print(f"已记录 {index} 个数据点")


if __name__ == "__main__":
    main()
