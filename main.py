import sys
import os

current_path = os.getcwd()
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path,'code'))

from ars import run_ars
import ray
import wandb

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_cpus', type=int, default=80)

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(_redis_address= local_ip + ':6379')
    # ray.init()



    args = parser.parse_args()
    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    wandb.init(project="PolicySpaceOptimization_new", entity="ofirnabati", config=args.__dict__)
    wandb.run.name = args.env_name + '_' + 'es'
    wandb.run.save()


    params = vars(args)
    run_ars(params)