#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简版：加载并运行 UR5e XML 模型
"""
import mujoco
import mujoco.viewer as mj_viewer
import time

# 替换为你的 XML 文件路径
XML_PATH = "D:/tutorial_mujoco/model/universal_robots_ur5e/ur5e_2.xml"

# 1. 加载模型
try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    print(f"✅ 成功加载模型：{XML_PATH}")
    print(f"   关节数：{model.njnt}")
    print(f"   连杆数：{model.nbody}")
except Exception as e:
    print(f"❌ 加载失败：{e}")
    exit(1)

# 2. 启动可视化
with mj_viewer.launch_passive(model, data) as viewer:
    # 设置初始视角（可选）
    viewer.cam.azimuth = -45
    viewer.cam.elevation = -30
    viewer.cam.distance = 1.5

    # 仿真循环
    while viewer.is_running():
        step_start = time.time()

        # 运行一步仿真（也可以暂停：注释掉这行，只可视化静态模型）
        mujoco.mj_step(model, data)

        # 刷新视图
        viewer.sync()

        # 控制帧率
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)