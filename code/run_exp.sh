set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3


##Hopper
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1  --max_timesteps 1000000 --eval_freq 5
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --seed 1 --max_timesteps 1000000 --eval_freq 5
#python code/ars.py --env_name Hopper-v3 --n_directions 8 --deltas_used 4 --step_size 0.01 --delta_std 0.025 --n_workers 50 --shift 1 --seed 2 --max_timesteps 1000000 --eval_freq 5

python code/ars_bandits.py  --env_name Hopper-v3 --n_directions 20 --n_bandit_directions 8 --deltas_used 4 --step_size 0.01 --shift 1 --delta_std 0.025 --n_workers 50 --max_timesteps 1000000 --eval_freq 5
python code/ars_bandits.py  --env_name Hopper-v3 --n_directions 20 --n_bandit_directions 8 --deltas_used 4 --step_size 0.01 --shift 1 --delta_std 0.025 --n_workers 50 --seed 1 --max_timesteps 1000000 --eval_freq 5
python code/ars_bandits.py  --env_name Hopper-v3 --n_directions 20 --n_bandit_directions 8 --deltas_used 4 --step_size 0.01 --shift 1 --delta_std 0.025 --n_workers 50 --seed 2 --max_timesteps 1000000 --eval_freq 5

##HalfCheetah
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 5000000
#python code/ars.py --env_name HalfCheetah-v3 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 5000000

python code/ars_bandits.py --n_bandit_directions 50 --env_name HalfCheetah-v3 --n_directions 100 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name HalfCheetah-v3 --n_directions 100 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name HalfCheetah-v3 --n_directions 100 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 5000000

#Ant
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
#python code/ars.py --env_name Ant-v3 --n_directions 60 --deltas_used 20 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000


python code/ars_bandits.py --n_bandit_directions 60 --env_name Ant-v3 --n_directions 120 --deltas_used 60 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 60 --env_name Ant-v3 --n_directions 120 --deltas_used 60 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 60 --env_name Ant-v3 --n_directions 120 --deltas_used 60 --step_size 0.015 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000



#SparseCheetah
python code/ars.py --env_name SparseHalfCheetah-v0 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 20000000
python code/ars.py --env_name SparseHalfCheetah-v0 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 20000000
python code/ars.py --env_name SparseHalfCheetah-v0 --n_directions 50 --deltas_used 4 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 20000000

python code/ars_bandits.py --n_bandit_directions 50 --env_name SparseHalfCheetah-v0 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 20000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name SparseHalfCheetah-v0 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 20000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name SparseHalfCheetah-v0 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.03 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 20000000

#Swimmer (v)

python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 2000000
python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 2000000
python code/ars.py --env_name Swimmer-v3 --n_directions 50 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 2000000


python code/ars_bandits.py --n_bandit_directions 50 --env_name Swimmer-v3 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --max_timesteps 2000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name Swimmer-v3 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 1 --max_timesteps 2000000
python code/ars_bandits.py --n_bandit_directions 50 --env_name Swimmer-v3 --n_directions 100 --deltas_used 50 --step_size 0.02 --delta_std 0.01 --n_workers 50 --shift 0  --eval_freq 2 --seed 2 --max_timesteps 2000000

#Walker2d
python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
python code/ars.py --env_name Walker2d-v3 --n_directions 40 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000


python code/ars_bandits.py --n_bandit_directions 40 --env_name Walker2d-v3 --n_directions 60 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 40 --env_name Walker2d-v3 --n_directions 60 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 1 --max_timesteps 5000000
python code/ars_bandits.py --n_bandit_directions 40 --env_name Walker2d-v3 --n_directions 60 --deltas_used 30 --step_size 0.03 --delta_std 0.025 --n_workers 50 --shift 1  --eval_freq 2 --seed 2 --max_timesteps 5000000


#Humanoid
python code/ars_bandits.py --n_bandit_directions 230 --env_name Humanoid-v3 --n_directions 460 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512 --max_timesteps 20000000
python code/ars_bandits.py --n_bandit_directions 230 --env_name Humanoid-v3 --n_directions 460 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512 --seed 1 --max_timesteps 20000000
python code/ars_bandits.py --n_bandit_directions 230 --env_name Humanoid-v3 --n_directions 460 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5 --eval_freq 2 --training_freq_network 230 --training_iter 512 --seed 2 --max_timesteps 20000000

python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --max_timesteps 20000000
python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --seed 1 --max_timesteps 20000000
python code/ars.py --env_name Humanoid-v3 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 50 --shift 5  --eval_freq 2 --seed 2 --max_timesteps 20000000
