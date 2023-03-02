set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0





#Humanoid
python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --max_timesteps 20000000
python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --max_timesteps 20000000 --seed 1
python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --max_timesteps 20000000 --seed 2
#python code/ars_bandits_exp_vae.py  --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512  --max_timesteps 20000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --memory_size 50000 --device 0 --batch_size 16 --explore_coeff 0.2
#python code/ars_bandits_exp_vae.py  --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512  --max_timesteps 20000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --memory_size 50000 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
#python code/ars_bandits_exp_vae.py  --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512  --max_timesteps 20000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --memory_size 50000 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2


##Walker2d
#python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
#python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
#python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000
#python code/ars_bandits_exp_vae.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2  --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2
#python code/ars_bandits_exp_vae.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2  --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 1
#python code/ars_bandits_exp_vae.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2  --max_timesteps 5000000 --discount 0.995 --gae-lambda 1.0 --average_first_state 64 --lambda_prior 0.1 --horizon 1024 --num_unroll_step 1024 --device 0 --batch_size 16 --explore_coeff 0.2 --seed 2

