set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0


##Hopper
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1  --max_timesteps 5000000 --eval_freq 5
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --seed 1 --max_timesteps 5000000 --eval_freq 5
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --seed 2 --max_timesteps 5000000 --eval_freq 5
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --eval_freq 5 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --eval_freq 5 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --eval_freq 5 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2

##HalfCheetah
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 5000000
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 5000000
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
python code/ars_bandits_exp_vae.py --n_bandit_directions 512 --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2

#Ant
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000
python code/ars_bandits_exp_vae.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2
python code/ars_bandits_exp_vae.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
python code/ars_bandits_exp_vae.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2


#Swimmer (v)

#python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 2000000
#python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 2000000
#python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 2000000
python code/ars_bandits_exp_vae.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2  --max_timesteps 2000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2
python code/ars_bandits_exp_vae.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2  --max_timesteps 2000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
python code/ars_bandits_exp_vae.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2  --max_timesteps 2000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2



python code/ars_bandits_exp_vae.py --env_name GaussGridWorld --n_directions 100 --deltas_used 100 --step_size 0.1 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 1000000 --rollout_length 20 --discount 1.0 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 32 --explore_coeff 1.0 --save_exp True --filter NoFilter --policy_type discrete
