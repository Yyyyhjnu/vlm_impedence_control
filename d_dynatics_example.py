import mujoco
import mujoco.viewer
import numpy as np
import time


def test_robot_torque(scene_xml_path):
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)

    # 2. 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("仿真已启动。")
        print("按 'Space' 键开始/暂停。")

        # 初始状态：让机械臂从 home 位置开始
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        step_count = 0
        while viewer.is_running():
            step_start = time.time()

            # --- 测试逻辑：在此处修改输入力矩 ---
            # 示例 1: 零力矩输入（机械臂会受重力掉落）
            #data.ctrl[:6] = 0.0

            # 示例 2: 仅重力补偿（机械臂应悬浮在空中）
            # 注意：只有在你的 XML 配置文件中使用了 motor/general 且 biastype="none" 时，
            # qfrc_bias 才完全代表重力+科氏力。
            #data.ctrl[:6] = data.qfrc_bias[:6]

            # 示例 3: 在重力补偿基础上，给第 2 个关节（大臂）加一点力矩看它摆动
            #data.ctrl[1] += 10.0

            # 3. 物理仿真步进
            mujoco.mj_step(model, data)

            # 4. 每隔一段时间打印一次数据
            if step_count % 100 == 0:
                # 获取末端位置 (wrist_3_link)
                ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
                ee_pos = data.xpos[ee_id]
                joint_qpos = data.qpos[:6]

                print("-" * 30)
                print(f"Time: {data.time:.2f}s")
                print(f"关节角度 (rad): {np.round(joint_qpos, 3)}")
                print(f"末端位置 (x,y,z): {np.round(ee_pos, 3)}")
                print(f"当前偏置力矩 (qfrc_bias): {np.round(data.qfrc_bias[:6], 2)}")

            step_count += 1
            viewer.sync()

            # 控制仿真频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    # 确保路径指向你保存场景 XML 的位置
    SCENE_XML = "model/universal_robots_ur5e/scene-30.xml"
    test_robot_torque(SCENE_XML)