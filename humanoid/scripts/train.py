import os
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

from humanoid.envs import *
from humanoid.utils import get_args, task_registry, export_policy_as_jit

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    export_path = os.path.join(ppo_runner.log_dir, 'exported', 'policies')
    
    run_name = args.run_name if args.run_name is not None else train_cfg.runner.run_name
    
    version = run_name if run_name else "1"
    export_filename = f"policy_{version}.pt"
    
    export_policy_as_jit(ppo_runner.alg.actor_critic, export_path, filename=export_filename)

if __name__ == '__main__':
    args = get_args()
    train(args)