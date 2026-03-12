import pybullet as p
import pybullet_data
import numpy as np
import os

# 初始化仿真环境
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 手动指定UR5e模型的绝对路径（替换为你实际的保存路径）
ur5e_urdf_path = r"D:\tutorial_mujoco\model\ur5e.urdf"
# 验证文件是否存在
if not os.path.exists(ur5e_urdf_path):
    raise FileNotFoundError(f"UR5e模型文件不存在：{ur5e_urdf_path}")

# 加载UR5e模型
ur5e_id = p.loadURDF(ur5e_urdf_path, [0, 0, 0], useFixedBase=True)

# 后续IK求解逻辑和方案1完全一致
end_effector_index = 6
target_pos = [0.5, 0.1, 0.3]
target_orn = p.getQuaternionFromEuler([0, np.pi, 0])

lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
upper_limits = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
joint_ranges = [2*np.pi]*6
rest_poses = [0, -np.pi/2, 0, 0, 0, 0]

joint_angles = p.calculateInverseKinematics(
    ur5e_id,
    end_effector_index,
    target_pos,
    target_orn,
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=joint_ranges,
    restPoses=rest_poses
)

print("UR5e IK求解结果：", joint_angles)

for i in range(6):
    p.resetJointState(ur5e_id, i, joint_angles[i])

while p.isConnected():
    p.stepSimulation()