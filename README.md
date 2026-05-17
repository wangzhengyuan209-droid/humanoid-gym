# Humanoid-Gym (Customized)

基于 [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) 框架的定制化人形机器人训练项目，使用 NVIDIA Isaac Gym 进行强化学习训练。

## 项目说明

本项目在原 Humanoid-Gym 框架基础上进行了定制修改，用于人形机器人的运动控制训练。主要修改包括：

- **P1 机器人配置**：针对自研 P1 型号人形机器人调整了关节 PD 增益、扭矩限制等参数
- **Sim2Sim 适配**：优化了从 Isaac Gym 到 Mujoco 的仿真迁移流程
- **策略部署**：支持训练后策略的导出和零样本迁移到真实机器人

## 安装

```bash
# 创建 Python 3.8 虚拟环境
conda create -n humanoid python=3.8
conda activate humanoid

# 安装 PyTorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装 Isaac Gym (Preview 4)
# 从 NVIDIA 官网下载后:
cd isaacgym/python && pip install -e .

# 安装本项目
cd humanoid-gym && pip install -e .
```

## 使用

```bash
cd humanoid

# 训练 PPO 策略
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# 评估策略并导出 JIT 模型
python scripts/play.py --task=humanoid_ppo --run_name v1

# Sim2Sim 迁移 (Mujoco)
python scripts/sim2sim.py --load_model /path/to/exported/policies/policy_1.pt
```

## 项目结构

```
humanoid-gym/
├── humanoid/
│   ├── envs/          # 环境配置 (base, p1 等)
│   ├── scripts/       # 训练、评估、sim2sim 脚本
│   ├── utils/         # 工具函数
│   └── logs/          # 训练日志和模型
├── resources/         # 机器人模型文件 (URDF/MJCF)
└── images/            # 演示图片
```

## 致谢

本项目的底层实现基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 项目，以及原始 [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) 框架。
