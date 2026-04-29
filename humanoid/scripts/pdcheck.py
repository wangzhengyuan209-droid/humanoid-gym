import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

from collections import deque

# -----------------------------
# Utils
# -----------------------------
def _name(obj_type, m: mujoco.MjModel, obj_id: int) -> str:
    n = mujoco.mj_id2name(m, obj_type, obj_id)
    return n if n is not None else f"<unnamed:{obj_id}>"

def find_freejoint_qpos_qvel_slices(m: mujoco.MjModel) -> Optional[Tuple[slice, slice]]:
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = m.jnt_qposadr[jid]
            dof_adr = m.jnt_dofadr[jid]
            return slice(qpos_adr, qpos_adr + 7), slice(dof_adr, dof_adr + 6)
    return None

def list_actuated_hinge_slide(m: mujoco.MjModel) -> Tuple[List[int], List[int], List[str]]:
    joint_ids, act_ids, joint_names = [], [], []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT: continue
        jid = int(m.actuator_trnid[a, 0])
        jt = m.jnt_type[jid]
        if jt not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE): continue
        joint_ids.append(jid)
        act_ids.append(a)
        joint_names.append(_name(mujoco.mjtObj.mjOBJ_JOINT, m, jid))
    return joint_ids, act_ids, joint_names

# -----------------------------
# Main Logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser("MuJoCo 单关节 PD 参数测试工具")

    parser.add_argument("--xml", type=str, required=True, help="XML文件路径")
    parser.add_argument("--kp", type=float, default=100.0, help="测试的 P 增益")
    parser.add_argument("--kv", type=float, default=1.0, help="测试的 D 增益")
    parser.add_argument("--joints", type=str, default="", help="要测试的关节名称（多个用逗号隔开），留空则只锁定不运动")
    
    parser.add_argument("--dt", type=float, default=0.001, help="仿真步长")
    parser.add_argument("--decimation", type=int, default=10, help="控制频率缩减（10代表100Hz控制）")
    parser.add_argument("--freq", type=float, default=0.5, help="正弦波频率(Hz)")
    parser.add_argument("--amp_scale", type=float, default=0.2, help="摆动幅度占关节限位的比例")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--plot_joint", type=str, default="")
    args = parser.parse_args()

    # 加载模型
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    m.opt.timestep = args.dt
    ctrl_dt = args.dt * args.decimation

    # 获取所有可控关节
    all_j_ids, all_a_ids, all_j_names = list_actuated_hinge_slide(m)
    all_qpos_idx = np.array([m.jnt_qposadr[jid] for jid in all_j_ids])
    all_qvel_idx = np.array([m.jnt_dofadr[jid] for jid in all_j_ids])

    print("\n--- [系统关节列表] ---")
    print(", ".join(all_j_names))
    print("----------------------\n")

    # 确定测试关节
    swing_names = [x.strip() for x in args.joints.split(",") if x.strip()]
    for name in swing_names:
        if name not in all_j_names:
            print(f"错误: 关节 '{name}' 不在可控关节列表中！")
            return
    
    swing_local_idxs = [all_j_names.index(n) for n in swing_names]

    # 初始化位置 (读取 XML 定义的姿态)
    mujoco.mj_forward(m, d)
    q0 = d.qpos[all_qpos_idx].copy()

    # 自动开启绘图：如果指定了关节测试且没手动选 plot_joint
    plot_target_name = args.plot_joint if args.plot_joint else (swing_names[0] if swing_names else "")
    plot_enabled = bool(plot_target_name)
    
    if plot_enabled:
        plot_i = all_j_names.index(plot_target_name)
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        t_buf, qd_buf, q_buf = deque(maxlen=500), deque(maxlen=500), deque(maxlen=500)
        line_des, = ax.plot([], [], 'r--', label=f"Target ({plot_target_name})")
        line_q, = ax.plot([], [], 'b-', label=f"Measured ({plot_target_name})")
        ax.set_title(f"PD Testing: Kp={args.kp}, Kv={args.kv}")
        ax.legend(loc="upper right"); ax.grid(True)

    # 锁定基座
    free_slices = find_freejoint_qpos_qvel_slices(m)
    base_q0 = d.qpos[free_slices[0]].copy() if free_slices else None

    print(f"正在测试关节: {swing_names if swing_names else '仅锁定全部关节'}")
    print(f"参数配置: Kp={args.kp}, Kv={args.kv}, 控制频率={1.0/ctrl_dt:.1f}Hz")

    counter = 0
    ctrl_step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < args.duration:
            step_start = time.time()

            # 强行重置基座姿态，防止机器人倒下影响观察
            if free_slices:
                d.qpos[free_slices[0]] = base_q0
                d.qvel[free_slices[1]] = 0

            # 控制循环 (按 decimation 降频)
            if counter % args.decimation == 0:
                t = ctrl_step * ctrl_dt
                q_target = q0.copy()
                
                # 计算选中关节的正弦目标
                for idx in swing_local_idxs:
                    jid = all_j_ids[idx]
                    # 获取限位范围
                    low, high = m.jnt_range[jid]
                    amp = args.amp_scale * (high - low) if m.jnt_limited[jid] else 0.3
                    q_target[idx] = q0[idx] + amp * np.sin(2 * np.pi * args.freq * t)

                # 执行 PD 计算
                q_curr = d.qpos[all_qpos_idx]
                v_curr = d.qvel[all_qvel_idx]
                torques = args.kp * (q_target - q_curr) - args.kv * v_curr
                
                # 应用力矩
                d.ctrl[all_a_ids] = torques

                # 更新绘图
                if plot_enabled:
                    t_buf.append(t); qd_buf.append(q_target[plot_i]); q_buf.append(q_curr[plot_i])
                    line_des.set_data(t_buf, qd_buf); line_q.set_data(t_buf, q_buf)
                    ax.relim(); ax.autoscale_view(); fig.canvas.draw_idle(); fig.canvas.flush_events()

                ctrl_step += 1

            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

            # 控制实时率
            time_until_next = m.opt.timestep - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()