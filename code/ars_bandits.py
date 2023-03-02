'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import ipdb
from gym.envs.mujoco import HalfCheetahEnv
import random
import time
import os
import numpy as np
import gym
import logz
import torch
import ray
import utils as utils
import optimizers as optimizers
from policies import *

from shared_noise import SharedNoiseTable, create_shared_noise
from neural_linear_gae2 import NeuralLinearPosteriorSampling
from gym.envs.registration import register
from half_cheetah_sparse import SparseHalfCheetahDirEnv
from storage import Storage
import wandb


register(id = 'SparseHalfCheetah-v0',entry_point = 'half_cheetah_sparse:SparseHalfCheetahDirEnv')

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        if env_name == 'SparseHalfCheetah-v0':
            self.env = SparseHalfCheetahDirEnv()
        else:
            self.env = gym.make(env_name)
        self.env.reset(seed=env_seed)
        # self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'nn':
            self.policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'],
                                               policy_params['ob_filter'],
                                               device='cpu:0')
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        # assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """

        obss    = []
        masks   = []
        actions = []
        rewards = []
        step_id   = []


        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            obss.append(ob)
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            masks.append(1 - done)
            actions.append(action)
            rewards.append(reward - shift)
            step_id.append(i)

            if done:
                    break

        # exps = utils.DictListObject()
        # exps.obs = obss
        # exps.action = np.concatenate(actions)
        # exps.reward = np.array(rewards)
        # exps.mask = np.array(masks)
        # exps.step = np.array(step_id)

        return total_reward, steps, obss, np.stack(actions), np.array(rewards), np.array(masks), np.array(step_id)

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False, idxes=None):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx, obss_arr, actions_arr, rewards_arr, masks_arr, step_id_arr= [], [], [], [], [], [], []
        steps = 0
        if idxes is not None:
            num_rollouts = len(idxes)

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                # reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                reward, r_steps, obss, actions, rewards, masks, step_id = self.rollout(shift = 0., rollout_length=self.rollout_length)
                rollout_rewards.append(reward)
                obss_arr.append(obss)
                actions_arr.append(actions)
                rewards_arr.append(rewards)
                masks_arr.append(masks)
                step_id_arr.append(step_id)

            else:
                if idxes is not None:
                    idx = idxes[i]
                    delta = self.deltas.get(idx, w_policy.size)
                else:
                    idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, pos_obss, pos_actions, pos_rewards, pos_masks, pos_step_id  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, neg_obss, neg_actions, neg_rewards, neg_masks, neg_step_id  = self.rollout(shift = shift)
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                obss_arr.append([pos_obss, neg_obss])
                actions_arr.append([pos_actions, neg_actions])
                rewards_arr.append([pos_rewards, neg_rewards])
                masks_arr.append([pos_masks, neg_masks])
                step_id_arr.append([pos_step_id, neg_step_id])

        return {'deltas_idx': deltas_idx,
                'rollout_rewards': rollout_rewards,
                'steps' : steps,
                "obss_arr": obss_arr,
                "actions_arr": actions_arr,
                "rewards_arr": rewards_arr,
                "masks_arr": masks_arr,
                "step_id_arr": step_id_arr}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=640,
                 num_bandit_deltas=320,
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123,
                 storage=None,
                 bandit_algo=None,
                 device='cpu'):

        # logz.configure_output_dir(logdir)
        # logz.save_params(params)
        
        env = gym.make(env_name)
        
        self.timesteps = 0
        self.rollouts = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.num_bandit_deltas = num_bandit_deltas
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.device = device

        self.bandit_algo = bandit_algo
        self.storage = storage
        self.average_first_state = params['average_first_state']
        self.horizon = params['horizon']
        self.discount = params['discount']
        self.eval_freq = params['eval_freq']
        self.max_timesteps = params['max_timesteps']

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]


        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'nn':
            self.policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'], policy_params['ob_filter'], device='cpu:0')
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_bandit_deltas
        else:
            num_deltas = num_rollouts

        if not evaluate:
            # idxes, deltas = self.deltas.get_deltas(self.w_policy.size, self.num_deltas)
            # deltas = (self.delta_std * deltas).reshape([self.num_deltas] + list(self.w_policy.shape))
            #
            # decison_set_pos = self.w_policy[np.newaxis] + deltas
            # decison_set_neg = self.w_policy[np.newaxis] - deltas
            # decison_set_pos = torch.tensor(decison_set_pos.reshape(self.num_deltas, -1), device=self.device, dtype=torch.float)
            # decison_set_neg = torch.tensor(decison_set_neg.reshape(self.num_deltas, -1), device=self.device, dtype=torch.float)
            # V_idx = torch.zeros(decison_set_pos.shape[0])
            # with torch.no_grad():
            #         # V_plus = 0
            #         # V_minus = 0
            #         for k in range(args.average_first_state):
            #             # first_state = self.storage.sample_state(i * self.horizon)
            #             first_state, frag_idx = self.storage.sample_random_state_and_fragment(self.horizon,
            #                                                                                   self.bandit_algo.fragments,
            #                                                                                   self.num_bandit_deltas)
            #             if first_state is None:
            #                 break
            #             first_state = torch.tensor(first_state, dtype=torch.float).to(device)
            #             first_state = torch.stack([first_state for _ in range(decison_set_pos.shape[0])])
            #             first_state = first_state.float()
            #             p1, best_idx, values_plus, _ = self.bandit_algo.action(decison_set_pos, first_state, fragment=frag_idx)
            #             p1, best_idx, values_minus, _ = self.bandit_algo.action(decison_set_neg, first_state, fragment=frag_idx)
            #             # V_plus += values_plus
            #             # V_minus += values_minus
            #             V = torch.stack([values_plus, values_minus])
            #             V, _ = torch.max(V, dim=0)
            #             _, sorted_indx = torch.sort(V)
            #             V_idx[sorted_indx[-self.num_bandit_deltas:]] += 1

            with torch.no_grad():
                for k in range(args.average_first_state):
                    idxes, deltas = self.deltas.get_deltas(self.w_policy.size, self.num_bandit_deltas)
                    deltas = (self.delta_std * deltas).reshape([self.num_bandit_deltas] + list(self.w_policy.shape))

                    decison_set_pos = self.w_policy[np.newaxis] + deltas
                    decison_set_neg = self.w_policy[np.newaxis] - deltas
                    decison_set_pos = torch.tensor(decison_set_pos.reshape(self.num_bandit_deltas, -1),
                                                   device=self.device,
                                                   dtype=torch.float)
                    decison_set_neg = torch.tensor(decison_set_neg.reshape(self.num_bandit_deltas, -1),
                                                   device=self.device,
                                                   dtype=torch.float)

                    Vpos, Vneg, emprical_return = 0, 0, 0
                    for _ in range(1):
                        first_state, frag_idx = self.storage.sample_random_state_and_fragment(self.horizon,
                                                                                              self.bandit_algo.fragments,
                                                                                              self.num_deltas * 2)
                        first_state = torch.tensor(first_state, dtype=torch.float).to(device)
                        first_state = torch.stack([first_state for _ in range(decison_set_pos.shape[0])])
                        first_state = first_state.float()
                        p1, best_idx, values_plus, _ = self.bandit_algo.action(decison_set_pos, first_state,
                                                                               fragment=frag_idx,
                                                                               ref_point=torch.tensor(
                                                                                   self.w_policy.reshape(-1),
                                                                                   device=self.device,
                                                                                   dtype=torch.float))
                        # p1, best_idx, values_plus, _ = self.bandit_algo.action(decison_set_pos, first_state, fragment=frag_idx)
                        p1, best_idx, values_minus, _ = self.bandit_algo.action(decison_set_neg, first_state,
                                                                                fragment=frag_idx)
                        Vpos += values_plus
                        Vneg += values_minus

                        # policy_id = ray.put(self.w_policy)
                        # num_rollouts = int(self.num_bandit_deltas / self.num_workers)
                        # rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                        #                                              idxes=None if evaluate else idxes[i * num_rollouts: (i + 1) * num_rollouts],
                        #                                              num_rollouts=num_rollouts if evaluate else None,
                        #                                              shift=self.shift,
                        #                                              evaluate=evaluate) for i, worker in enumerate(self.workers)]
                        # results_two = ray.get(rollout_ids_one)
                        # result_rewards = []
                        # for result in results_two:
                        #     result_rewards += result['rollout_rewards']
                        # result_rewards = np.array(result_rewards, dtype=np.float64)
                        # emprical_return += result_rewards
                        # print(f'Compare returns: {np.mean(np.abs(result_rewards - V.cpu().numpy()))}')

                    V = torch.stack([Vpos / 10.0, Vneg / 10.0], dim=1)
                    # emprical_return = emprical_return / 10.0
                    V_max, V_max_idx_1 = torch.max(V, dim=1)
                    V = V.cpu().numpy()


            _ , es_set_idxes = torch.sort(V_max)
            es_set_idxes = es_set_idxes[-self.num_bandit_deltas:]
            deltas = deltas[es_set_idxes]
            idxes  = idxes[es_set_idxes]
            if num_deltas == 1:
                idxes = [idxes]


        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        if num_rollouts > 0:
            rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     idxes = None if evaluate else idxes[i*num_rollouts : (i+1)*num_rollouts],
                                                     num_rollouts = num_rollouts if evaluate else None,
                                                     shift = self.shift,
                                                     evaluate=evaluate) for i,worker in enumerate(self.workers)]
        else:
            rollout_ids_one = []
            results_one = []

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1 if evaluate else None,
                                                 idxes = None if evaluate else [idxes[self.num_workers * num_rollouts + i]],
                                                 shift = self.shift,
                                                 evaluate=evaluate) for i, worker in enumerate(self.workers[:(num_deltas % self.num_workers)])]

        # gather results
        if num_rollouts > 0:
            results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []


        loss = []
        rollout_len = 0
        for result in results_one:
            if not evaluate:
                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                for j in range(num_rollouts):
                    delta = self.deltas.get(result['deltas_idx'][j], self.w_policy.size)
                    delta = self.delta_std * delta
                    exp_pos = utils.DictListObject()
                    exp_pos.obs    = result['obss_arr'][j][0]
                    exp_pos.action = result['actions_arr'][j][0]
                    exp_pos.reward = result['rewards_arr'][j][0]
                    exp_pos.mask   = result['masks_arr'][j][0]
                    exp_pos.step   = result['step_id_arr'][j][0]
                    if self.params['filter'] == 'MeanStdFilter':
                        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                        update_par = np.concatenate([self.w_policy.reshape(-1) + delta, w[1], w[2]])
                    else:
                        update_par = self.w_policy.reshape(-1) + delta
                    loss_pos = self.bandit_algo.update(exp_pos, update_par)
                    # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                    if loss_pos is not None:
                        loss.append(loss_pos)


                    exp_neg = utils.DictListObject()
                    exp_neg.obs    = result['obss_arr'][j][1]
                    exp_neg.action = result['actions_arr'][j][1]
                    exp_neg.reward = result['rewards_arr'][j][1]
                    exp_neg.mask   = result['masks_arr'][j][1]
                    exp_neg.step   = result['step_id_arr'][j][1]
                    if self.params['filter'] == 'MeanStdFilter':
                        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                        update_par = np.concatenate([self.w_policy.reshape(-1) - delta, w[1], w[2]])
                    else:
                        update_par = self.w_policy.reshape(-1) - delta
                    loss_neg = self.bandit_algo.update(exp_neg, update_par)
                    # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                    if loss_neg is not None:
                        loss.append(loss_neg)

            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']


        for result in results_two:
            if not evaluate:

                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                delta = self.deltas.get(result['deltas_idx'][0], self.w_policy.size)
                delta = self.delta_std * delta
                exp_pos = utils.DictListObject()
                exp_pos.obs = result['obss_arr'][0][0]
                exp_pos.action = result['actions_arr'][0][0]
                exp_pos.reward = result['rewards_arr'][0][0]
                exp_pos.mask = result['masks_arr'][0][0]
                exp_pos.step = result['step_id_arr'][0][0]
                if self.params['filter'] == 'MeanStdFilter':
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                    update_par = np.concatenate([self.w_policy.reshape(-1) + delta, w[1], w[2]])
                else:
                    update_par = self.w_policy.reshape(-1) + delta
                loss_pos = self.bandit_algo.update(exp_pos, update_par)
                # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                if loss_pos is not None:
                    loss.append(loss_pos)

                exp_neg = utils.DictListObject()
                exp_neg.obs = result['obss_arr'][0][1]
                exp_neg.action = result['actions_arr'][0][1]
                exp_neg.reward = result['rewards_arr'][0][1]
                exp_neg.mask = result['masks_arr'][0][1]
                exp_neg.step = result['step_id_arr'][0][1]
                if self.params['filter'] == 'MeanStdFilter':
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                    update_par = np.concatenate([self.w_policy.reshape(-1) - delta, w[1], w[2]])
                else:
                    update_par = self.w_policy.reshape(-1) - delta
                loss_neg = self.bandit_algo.update(exp_neg, update_par)
                # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                if loss_neg is not None:
                    loss.append(loss_neg)

            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)

        self.timesteps += rollout_len
        if not evaluate:
            print(f'Average rollout length:{rollout_len / (num_deltas * 2)}')
            print('Maximum reward of collected rollouts:', rollout_rewards.max())
        else:
            print('EVAL: Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_bandit_deltas:
            self.deltas_used = self.num_bandit_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * (1 - (self.deltas_used / self.num_bandit_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, loss
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat, loss = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return loss

    def train(self, num_iter):

        loss = 0
        start = time.time()
        # for i in range(num_iter):
        i = 0
        while self.timesteps <= self.max_timesteps:

            t1 = time.time()
            loss1 = self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')
            self.rollouts += self.num_bandit_deltas * 2

            if len(loss1) > 0:
                loss = np.mean(loss1)
            # record statistics every 10 iterations
            if ((i + 1) % self.eval_freq == 0):
                
                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)
                
                print(sorted(self.params.items()))
                # logz.log_tabular("Time", time.time() - start)
                # logz.log_tabular("Iteration", i + 1)
                # logz.log_tabular("AverageReward", np.mean(rewards))
                # logz.log_tabular("StdRewards", np.std(rewards))
                # logz.log_tabular("MaxRewardRollout", np.max(rewards))
                # logz.log_tabular("MinRewardRollout", np.min(rewards))
                # logz.log_tabular("timesteps", self.timesteps)
                # logz.log_tabular("rollout_num", self.rollouts)
                # logz.dump_tabular()

                wandb.log({"iteration": i + 1,
                           "value": np.mean(rewards),
                           "value std": np.std(rewards),
                           "value max": np.max(rewards),
                           "value min": np.min(rewards),
                           "global_step": self.timesteps,
                           "rollout_num": self.rollouts,
                           "mean_loss": loss,
                           "ls_bandit_val": self.bandit_algo.ls_values.mean().item(),
                           "network_val": self.bandit_algo.est_vals.mean().item(),
                           "ucb mean": self.bandit_algo.ucb.mean().item(),
                           "ucb std": self.bandit_algo.ucb.std().item(),
                           "overall ucb": self.bandit_algo.ls_values.mean().item() + 10.0 * self.bandit_algo.ucb.mean().item(),
                           "radius": self.bandit_algo.R
                           })


                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
            i += 1
        return 

def run_ars(args):

    params = vars(args)
    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type': params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    if policy_params['type'] == 'linear':
        policy = LinearPolicy(policy_params)
    elif policy_params['type'] == 'nn':
        policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'],
                                           policy_params['ob_filter'],
                                           device='cpu:0')
    else:
        raise NotImplementedError

    if params['filter'] == 'MeanStdFilter':
        w = policy.get_weights_plus_stats()
        context_dim = np.concatenate([w[0].reshape(-1), w[1], w[2]]).size
    else:
        context_dim = policy.get_weights().size
    args.context_dim = context_dim
    args.obs_dim = ob_dim

    storage = Storage(config=args)
    bandit_algo = NeuralLinearPosteriorSampling(storage, device, args)

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     num_bandit_deltas = params['n_bandit_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'],
                     bandit_algo=bandit_algo,
                     storage=storage,
                     device=params['device'])
        
    ARS.train(params['n_iter'])
       
    return 




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--max_timesteps', '-max_t', type=int, default=5e6)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--n_bandit_directions', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--average_first_state', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=10)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_cpus', type=int, default=80)
    parser.add_argument('--device', type=int, default=0)

    #bandits args
    parser.add_argument("--discount", type=float, default=0.995,
                        help="discount factor (default: 0.9996)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--use_target_network", type=bool, default=True)
    parser.add_argument("--policy_history_set_size", type=int, default=3)
    parser.add_argument("--num_unroll_steps", type=int, default=64)
    parser.add_argument("--method", type=str, default='ucb')
    parser.add_argument("--target_model_update", type=int, default=500)
    parser.add_argument("--a0", type=float, default=6)
    parser.add_argument("--b0", type=float, default=1000)
    parser.add_argument("--ucb_coeff", type=float, default=10.0)
    parser.add_argument("--lambda_prior", type=float, default=100.0)
    parser.add_argument("--memory_size", type=int, default=500)
    parser.add_argument("--layers_size", type=int, default=[2048, 2048])
    parser.add_argument("--lr_step_size", type=int, default=100000)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--training_freq_network", type=int, default=50)
    parser.add_argument("--training_iter", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument("--state_based_value", type=bool, default=False)

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(_redis_address= local_ip + ':6379')
    # ray.init()

    args = parser.parse_args()
    args.training_freq_network = args.n_directions * 2
    args.training_iter = args.n_directions * 6 # args.n_bandit_directions * 3
    args.target_model_update = args.training_iter

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.fragments = args.rollout_length // args.horizon
    # if args.rollout_length % args.horizon > 0:
    #     args.fragments += 1
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    wandb.init(project="PolicySpaceOptimization_new", entity="ofirnabati", config=args.__dict__)
    wandb.run.name = args.env_name + '_' + 'neural_es'
    if args.policy_type == 'nn':
        wandb.run.name = wandb.run.name + '_nn'
    wandb.run.save()



    args.device = device
    run_ars(args)