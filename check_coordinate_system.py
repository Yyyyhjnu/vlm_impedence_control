"""
检查 ikpy 链和 MuJoCo 之间的坐标系差异
"""
import mujoco
import ikpy.chain
import numpy as np
import transforms3d as tf

def analyze_coordinate_transforms():
    """分析 URDF 和 MuJoCo 中的坐标系变换"""
    
    print("=" * 80)
    print("坐标系变换分析")
    print("=" * 80)
    
    # 1. 检查 URDF 中的变换
    print("\n1. URDF 中的坐标系变换:")
    print("   - global_transform: base_link -> transformed_base_link")
    print("     旋转: rpy='0 0 3.1415926535' (绕Z轴旋转180度)")
    print("     这意味着 transformed_base_link 相对于 base_link 旋转了180度")
    print("   - base_link-base_fixed_joint: base_link -> base")
    print("     旋转: rpy='0 0 -3.141592653589793' (绕Z轴旋转-180度)")
    print("   - shoulder_pan_joint 的父链接: transformed_base_link")
    
    # 2. 检查 MuJoCo 中的变换
    print("\n2. MuJoCo (ur5e.xml) 中的坐标系变换:")
    print("   - base body: quat='0 0 0 -1' (相当于绕某个轴旋转180度)")
    print("   - 这可能导致坐标系与 URDF 不一致")
    
    # 3. 检查 ikpy 链的基座
    print("\n3. ikpy 链的基座:")
    print("   - ikpy 通常使用第一个活动链接的父链接作为基座")
    print("   - 如果使用 transformed_base_link，坐标系可能与 MuJoCo 不同")
    
    # 4. 检查末端执行器定义
    print("\n4. 末端执行器定义:")
    print("   - URDF 中: ee_link 相对于 wrist_3_link 的偏移")
    print("     xyz='0.0 0.1 0.0', rpy='0.0 0.0 1.5707963267948966' (绕Z轴旋转90度)")
    print("   - tool0 相对于 wrist_3_link:")
    print("     xyz='0 0.1 0', rpy='-1.5707963267948966 0 0' (绕X轴旋转-90度)")
    
    print("\n" + "=" * 80)
    print("潜在问题:")
    print("=" * 80)
    print("1. ikpy 链可能使用 transformed_base_link 作为基座，而 MuJoCo 使用 base")
    print("2. 两者之间可能有 180 度的旋转差异")
    print("3. 这会导致 X 和 Y 轴反向，但 Z 轴可能保持一致")
    print("4. 末端执行器的偏移定义不同（ee_link vs tool0）")
    
    return True


def test_coordinate_consistency():
    """测试坐标系一致性"""
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene-30.xml')
    data = mujoco.MjData(model)
    
    # 加载 ikpy 链
    chain = ikpy.chain.Chain.from_urdf_file(
        "model/ur5e_orig.urdf",
        active_links_mask=[False, False] + [True] * 6 + [False]
    )
    
    print("\n" + "=" * 80)
    print("坐标系一致性测试")
    print("=" * 80)
    
    # 测试1: 检查基座位置
    print("\n测试1: 基座位置")
    print("  ikpy 链的基座链接:", chain.links[0].name if hasattr(chain.links[0], 'name') else "未知")
    
    # 测试2: 使用零关节角度测试
    print("\n测试2: 零关节角度位置")
    zero_joints = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # ikpy 正运动学
    fk_ikpy = chain.forward_kinematics(zero_joints)
    pos_ikpy = fk_ikpy[:3, 3]
    print(f"  ikpy 计算位置: {np.round(pos_ikpy, 4)}")
    
    # MuJoCo 正运动学
    data.qpos[:6] = zero_joints[2:-1]
    mujoco.mj_forward(model, data)
    
    try:
        wrist3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        if wrist3_id >= 0:
            pos_mujoco = data.xpos[wrist3_id]
            print(f"  MuJoCo wrist_3_link: {np.round(pos_mujoco, 4)}")
            diff = pos_ikpy - pos_mujoco
            print(f"  位置差异: {np.round(diff, 4)}")
            print(f"  差异模长: {np.round(np.linalg.norm(diff), 4)} 米")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 测试3: 检查链的链接顺序
    print("\n测试3: ikpy 链的链接信息")
    print(f"  链长度: {len(chain.links)}")
    for i, link in enumerate(chain.links):
        if hasattr(link, 'name'):
            print(f"  链接 {i}: {link.name}")
        else:
            print(f"  链接 {i}: (无名称)")
    
    # 测试4: 检查末端链接
    print("\n测试4: 末端链接")
    if len(chain.links) > 0:
        last_link = chain.links[-1]
        if hasattr(last_link, 'name'):
            print(f"  末端链接名称: {last_link.name}")
        else:
            print(f"  末端链接: (无名称)")
    
    return {
        'ikpy_pos': pos_ikpy,
        'mujoco_pos': pos_mujoco if 'pos_mujoco' in locals() else None
    }


if __name__ == "__main__":
    analyze_coordinate_transforms()
    test_coordinate_consistency()
