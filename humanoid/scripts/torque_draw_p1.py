import math
import random
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import P1Cfg 
import torch
import matplotlib.pyplot as plt
import os

# --- 核心配置：从你提供的 XML 资料中提取的力矩范围 ---
# 顺序: Hip_Roll, Hip_Pitch, Hip_Yaw, Knee, Ankle_Pitch, Ankle_Roll (L & R)
P1_TORQUE_LIMITS = np.array([
    120, 120, 60, 60, 34, 34,  # 左腿
    120, 120, 60, 60, 34, 34   # 右腿
], dtype=np.double)

class cmd:
    vx = 0.6
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) 
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    if not os.path.exists(cfg.sim_config.mujoco_model_path):
        print(f"Error: Mujoco model file not found at {cfg.sim_config.mujoco_model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    for _ in range(40):
        mujoco.mj_step(model, data)

    q, dq, quat, v, omega, gvec = get_obs(data)
    q = q[-cfg.env.num_actions:] 
    dq = dq[-cfg.env.num_actions:] 

    eu_ang = quaternion_to_euler_array(quat)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi 

    init_phase_offset = random.uniform(0, 1.0)
    action = np.zeros(cfg.env.num_actions, dtype=np.float32)

    obs0 = np.zeros((1, cfg.env.num_single_obs), dtype=np.float32)
    obs0[0, 0] = math.sin(2 * math.pi * init_phase_offset)
    obs0[0, 1] = math.cos(2 * math.pi * init_phase_offset)
    obs0[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
    obs0[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
    obs0[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
    obs0[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
    obs0[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
    obs0[0, 29:41] = action; obs0[0, 41:44] = omega; obs0[0, 44:47] = eu_ang
    obs0 = np.clip(obs0, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

    hist_obs = deque([obs0.copy() for _ in range(cfg.env.frame_stack)])

    viewer = mujoco_viewer.MujocoViewer(model, data)
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)

    plot_data = {'time': [], 'torques': [], 'velocities': []}
    joint_names = [
        'L_Hip_Roll', 'L_Hip_Pitch', 'L_Hip_Yaw', 'L_Knee', 'L_Ankle_Pitch', 'L_Ankle_Roll',
        'R_Hip_Roll', 'R_Hip_Pitch', 'R_Hip_Yaw', 'R_Knee', 'R_Ankle_Pitch', 'R_Ankle_Roll'
    ]

    count_lowlevel = 0
    sim_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    for _ in tqdm(range(sim_steps), desc="Simulating..."):
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]; dq = dq[-cfg.env.num_actions:] 

        if count_lowlevel % cfg.sim_config.decimation == 0:
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            phase = (count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + init_phase_offset) % 1.0

            obs = np.zeros((1, cfg.env.num_single_obs), dtype=np.float32)
            obs[0, 0] = math.sin(2 * math.pi * phase); obs[0, 1] = math.cos(2 * math.pi * phase)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action; obs[0, 41:44] = omega; obs[0, 44:47] = eu_ang
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs); hist_obs.popleft()
            policy_input = np.zeros((1, cfg.env.num_observations), dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs:(i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale 

        target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -P1_TORQUE_LIMITS, P1_TORQUE_LIMITS)
        data.ctrl = tau 

        if count_lowlevel % 10 == 0:
            plot_data['time'].append(count_lowlevel * cfg.sim_config.dt)
            plot_data['torques'].append(tau.copy())
            plot_data['velocities'].append(dq.copy()) 

        mujoco.mj_step(model, data); viewer.render(); count_lowlevel += 1

    viewer.close()

    # 数据过滤（前 1.5 秒）
    final_time = np.array(plot_data['time'])
    plot_mask = final_time >= 1.5
    plot_time = final_time[plot_mask]
    plot_torques = np.array(plot_data['torques'])[plot_mask]

    # --- 绘图逻辑优化：显式显示 Limit 刻度 ---
    fig_torque, axs_torque = plt.subplots(6, 2, figsize=(12, 20), sharex=True)
    fig_torque.suptitle('P1 Motor Torque Outputs (Nm) - Ticks at Limits', fontsize=16)

    for i in range(12):
        row, col = i % 6, i // 6
        limit = P1_TORQUE_LIMITS[i]
        
        axs_torque[row, col].plot(plot_time, plot_torques[:, i], color='blue', lw=1.2)
        axs_torque[row, col].set_title(f'{joint_names[i]} (Limit: ±{limit})')
        axs_torque[row, col].grid(True, alpha=0.3)
        
        # 强制设置 Y 轴刻度，包含真实的上下限
        axs_torque[row, col].set_yticks([-limit, 0, limit])
        # 强制设置 Y 轴范围，留出 10% 的边缘防止曲线贴边
        axs_torque[row, col].set_ylim(-limit * 1.15, limit * 1.15)
        
        # 绘制明显的红色虚线作为边界
        axs_torque[row, col].axhline(y=limit, color='red', linestyle='--', linewidth=1.5)
        axs_torque[row, col].axhline(y=-limit, color='red', linestyle='--', linewidth=1.5)
        
        if col == 0: axs_torque[row, col].set_ylabel('Nm')

    axs_torque[5, 0].set_xlabel('Time (s)'); axs_torque[5, 1].set_xlabel('Time (s)')
    fig_torque.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_torque.savefig('p1_torque_labeled_limits.png')
    
    print("✅ 绘图完成。现在 Y 轴刻度会精确显示 ±60, ±120 或 ±34。")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, required=True)
    args = parser.parse_args()

    class Sim2simCfg(P1Cfg): 
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/p1/mjcf/scene.xml' 
            sim_duration = 10.0; dt = 0.001; decimation = 10     
        class robot_config:
            kps = np.array([200, 350, 200, 350, 20, 20, 200, 350, 200, 350, 20, 20], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 1, 1, 10, 10, 10, 10, 1, 1], dtype=np.double)
            tau_limit = P1_TORQUE_LIMITS 

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())