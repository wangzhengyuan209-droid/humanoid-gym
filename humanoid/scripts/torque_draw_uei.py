import math
import random
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import UEICfg
import torch
import matplotlib.pyplot as plt

class cmd:
    vx = 0.0
    vy = 0.4
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
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # 先运行几步让传感器数据稳定
    for _ in range(40):
        mujoco.mj_step(model, data)

    # 获取第一次真实观测并填充历史队列
    q, dq, quat, v, omega, gvec = get_obs(data)
    q = q[-cfg.env.num_actions:]
    dq = dq[-cfg.env.num_actions:]

    eu_ang = quaternion_to_euler_array(quat)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi

    # 随机化初始相位偏移 (与训练环境一致)
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
    obs0[0, 29:41] = action
    obs0[0, 41:44] = omega
    obs0[0, 44:47] = eu_ang
    obs0 = np.clip(obs0, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

    # 用第一个真实观测填充历史队列
    hist_obs = deque([obs0.copy() for _ in range(cfg.env.frame_stack)])

    viewer = mujoco_viewer.MujocoViewer(model, data)
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)

    # 存储力矩数据
    plot_data = {'time': [], 'torques': []}
    joint_names = [
        'L_Leg_Roll', 'L_Leg_Pitch', 'L_Leg_Yaw', 'L_Knee', 'L_Ankle_Pitch', 'L_Ankle_Roll',
        'R_Leg_Roll', 'R_Leg_Pitch', 'R_Leg_Yaw', 'R_Knee', 'R_Ankle_Pitch', 'R_Ankle_Roll'
    ]

    count_lowlevel = 0
    sim_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    for _ in tqdm(range(sim_steps), desc="Simulating..."):
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            phase = (count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time + init_phase_offset) % 1.0

            obs = np.zeros((1, cfg.env.num_single_obs), dtype=np.float32)
            obs[0, 0] = math.sin(2 * math.pi * phase)
            obs[0, 1] = math.cos(2 * math.pi * phase)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros((1, cfg.env.num_observations), dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs:(i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale

        target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau

        # 记录力矩 (每 10 步采样一次)
        if count_lowlevel % 10 == 0:
            plot_data['time'].append(count_lowlevel * cfg.sim_config.dt)
            plot_data['torques'].append(tau.copy())

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

    # 绘图
    print("\n正在生成电机力矩曲线图...")
    times = np.array(plot_data['time'])
    torques = np.array(plot_data['torques'])
    fig, axs = plt.subplots(6, 2, figsize=(12, 15), sharex=True)
    fig.suptitle('Motor Torque Outputs (Nm)', fontsize=16)

    for i in range(12):
        row, col = i % 6, i // 6
        axs[row, col].plot(times, torques[:, i], color='blue', alpha=0.8)
        axs[row, col].set_title(joint_names[i])
        axs[row, col].set_ylabel('Nm')
        axs[row, col].grid(True, alpha=0.3)
        # 标出力矩极限线
        limit = cfg.robot_config.tau_limit[i]
        axs[row, col].axhline(y=limit, color='r', linestyle='--', alpha=0.5)
        axs[row, col].axhline(y=-limit, color='r', linestyle='--', alpha=0.5)

    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('torque_curves.png')
    print("✅ 力矩曲线图已保存为 'torque_curves.png'")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, help='Path to policy_v5.pt')
    args = parser.parse_args()

    class Sim2simCfg(UEICfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/V1/mjcf/scene.xml'
            sim_duration = 10.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([250, 350, 250, 350, 30, 30, 250, 350, 250, 350, 30, 30], dtype=np.double)
            kds = np.array([7, 10, 7, 10, 1, 1, 7, 10, 7, 10, 1, 1], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())