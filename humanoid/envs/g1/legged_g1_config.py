from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LeggedG1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        episode_length_s = 60
        num_envs = 512

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'  # 启用高度场地形，而不是平面
        curriculum = True  # 启用课程学习
        # 创建一个明显不同的地形分布 - 90%都是楼梯！
        terrain_proportions = [0.0, 0.0, 0.7, 0.2, 0.1]  # [平滑斜坡0%, 粗糙斜坡0%, 上楼梯70%, 下楼梯20%, 离散障碍物10%]
        # 地形参数
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # 增加到10个难度等级，提供更丰富的课程学习
        num_cols = 15  # 保持地形类型数量
        max_init_terrain_level = 0  # 强制所有机器人从最简单的地形开始（第0行）
        # 地形摩擦力设置
        static_friction = 1.2   # 提高静摩擦力，让楼梯更好抓地
        dynamic_friction = 1.0  # 动摩擦力保持不变
        restitution = 0.1       # 稍微增加弹性，减少硬着陆
        # 测量点设置 - 用于感知地形高度
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    # 添加commands配置，让机器人沿着难度递增方向前进
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.  # 命令重采样时间间隔[秒]
        heading_command = False  # 不使用朝向命令，直接控制角速度
        class ranges:
            # 关键：在课程学习地形中，X方向是难度递增方向！
            # 机器人需要沿着+X方向前进，从简单地形走向复杂地形
            lin_vel_x = [0.8, 1.5]   # 沿+X方向前进（难度递增方向），速度0.8-1.5 m/s
            lin_vel_y = [0, 0]  # 允许轻微的左右调整，避免过于僵硬
            ang_vel_yaw = [0, 0] # 允许轻微转向调整，保持前进方向
            heading = [-3.14, 3.14]  # 保持默认朝向范围（虽然不会使用）


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.4]  # 缩小摩擦力范围，提高最小值，让机器人更稳定
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            # 增大全局 GPU 接触对和缓冲区，避免三角网格+多机器人导致的溢出
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 10

class LeggedG1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 100000
        run_name = ''
        experiment_name = 'legged_g1'
