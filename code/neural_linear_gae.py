"""Thompson Sampling with linear posterior over a learnt deep representation."""

import ipdb
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from neural_bandit_model import NeuralBanditModel

class NeuralLinearPosteriorSampling:
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, storage,  device, hparams):

    self.hparams = hparams
    self.storage = storage
    # self.latent_dim = self.hparams.context_dim + self.hparams.obs_dim
    self.latent_dim = self.hparams.layers_size[-1]
    self.param_dim=self.latent_dim
    self.context_dim = self.hparams.context_dim
    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior
    self.device = device
    self.fragments = 1
    self.horizon = hparams.horizon
    self.dtype = torch.double
    self.ucb_coeff = hparams.ucb_coeff


    self.mu = [torch.zeros(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]
    self.f  = [torch.zeros(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]
    self.yy = [0 for _ in range(self.fragments)]

    # self.cov = [(1.0 / self.lambda_prior) * torch.eye(self.param_dim, device=device) for _ in range(self.fragments)]

    self.precision = [self.lambda_prior * torch.eye(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.fragments)]
    self.b = [self._b0 for _ in range(self.fragments)]

    # Regression and NN Update Frequency
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.training_steps = 0

    self.data_h = storage
    self.model = NeuralBanditModel(hparams).to(self.device)

    self.target_model = NeuralBanditModel(hparams).to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    for param in self.target_model.parameters():
        param.requires_grad = False
    self.target_model.eval()

    self.method = hparams.method
    self.batch_data_number = 100

    self.ucb = 0
    self.ls_values = 0
    self.R = 0

    #Model learning
    # self.loss_fn = torch.nn.MSELoss()
    self.loss_fn = torch.nn.L1Loss()
    self.lr = hparams.initial_lr
    self.batch_size = hparams.batch_size
    self.training_iter = hparams.training_iter
    self.device = device
    self.lr_decay_rate = hparams.lr_decay_rate
    self.lr_step_size = hparams.lr_step_size
    self.max_grad_norm = hparams.max_grad_norm
    self.optimizer_name = hparams.optimizer
    self.gamma = hparams.discount
    self.gae_lambda = hparams.gae_lambda
    self.target_model_update = hparams.target_model_update
    self.soft_target_tau = 1e-1
    self.num_unroll_steps = hparams.num_unroll_steps
    self.stacked_observations = 1



    if self.optimizer_name == 'Adam':
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-8)
    elif self.optimizer_name == 'RMSprop':
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-8)
    else:
        raise ValueError('optimizer name is unkown')
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_decay_rate)


  def add_bandit_fragment(self):
      self.mu.append(torch.zeros(self.param_dim, device=self.device, dtype=self.dtype))
      self.f.append(torch.zeros(self.param_dim, device=self.device, dtype=self.dtype))
      self.yy.append(0)

      # self.cov.append((1.0 / self.lambda_prior) * torch.eye(self.param_dim, device=self.device))
      self.precision.append(self.lambda_prior * torch.eye(self.param_dim, device=self.device, dtype=self.dtype))

      self.a.append(self._a0)
      self.b.append(self._b0)


  def soft_update_from_to(self, source, target):
      for target_param, param in zip(target.parameters(), source.parameters()):
          target_param.data.copy_(
              target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
          )


  def action(self, decison_set, obs, fragment, ref_point=None):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    # if self.t < self.hparams.initial_pulls:
    #   return  torch.randn(self.context_dim).to(self.device), torch.zeros([]), torch.zeros([])

    self.model.eval()
    set_size = decison_set.shape[0]
    with torch.no_grad():
      # network_values, decison_set_latent = self.model(obs, decison_set)
      # decison_set_latent = self.model.encode(obs, decison_set)
      est_vals, decison_set_latent = self.model(obs, decison_set)
      self.est_vals = est_vals
      decison_set_latent = decison_set_latent.to(self.dtype)
      if ref_point is not None:
        ref_latent = self.model.encode(obs[:1], ref_point.unsqueeze(0))
        R = decison_set_latent - ref_latent
        self.R = R.norm(dim=-1).mean().item()

    # Sample sigma2, and beta conditional on sigma2
    # if self.b > 0:
    #   sigma2_s = self.b * invgamma.rvs(self.a)
    # else:
    #   print('Warning: parameter b is negative!')
    #   sigma2_s = 1e-4
    sigma2_s = 10.0

    if self.method == 'ucb':
        d = self.latent_dim
        self.ucb = torch.sqrt(torch.sum(torch.linalg.solve((1/sigma2_s) * self.precision[fragment],decison_set_latent.T).T * decison_set_latent, dim=1))
        self.ls_values = decison_set_latent @ self.mu[fragment]
        values = self.ls_values + self.ucb_coeff * self.ucb
        if torch.isnan(values).sum() > 0:
          print('cov is not PSD.. using default setting')
          ucb_default = torch.sqrt(torch.sum(torch.linalg.solve((1 / sigma2_s) * torch.eye(d, device=self.device, dtype=self.dtype),decison_set_latent.T).T * decison_set_latent, dim=1))
          values = self.ls_values + self.ucb_coeff * ucb_default
    # elif self.method == 'es':
    #     values = value_samples
    #     values = network_values
    elif self.method == 'ts':
        try:
          w_dist = MultivariateNormal(self.mu[fragment] ,precision_matrix= (1 / sigma2_s) * self.precision[fragment])
          w = w_dist.sample()
        except:
          # Sampling could fail if covariance is not positive definite
          d = self.param_dim
          w_dist = MultivariateNormal(torch.zeros(d, device=self.device), torch.eye(d, device=self.device))
          w = w_dist.sample()
        # decison_set_latent = decison_set_latent[R <= self.decison_set_radius]
        values =  torch.matmul(decison_set_latent,w)
    else:
      raise ValueError('method is unknown')

    best_arm = decison_set[values.argmax()]
    best_arm_index = values.argmax()
    # best_value = values.max()

    return best_arm,  best_arm_index, values, values.std()



  def compute_returns_and_advantage(self, masks, rewards, values) -> None:
      """
      Post-processing step: compute the lambda-return (TD(lambda) estimate)
      and GAE(lambda) advantage.

      Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
      to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
      where R is the sum of discounted reward with value bootstrap
      (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

      The TD(lambda) estimator has also two special cases:
      - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
      - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

      For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

      :param last_values: state value estimation for the last step (one for each env)
      :param dones: if the last step was a terminal step (one bool for each env).
      """
      flag = (masks[:,-1] == 0)
      values = values.clone()
      values[torch.where(flag)[0],-1] = rewards[torch.where(flag)[0],-1].float()
      # if masks.shape[1] == 1:
      #     return values

      masks = masks[:,:-1]
      rewards = rewards[:,:-1]
      advantages = torch.zeros_like(rewards)
      traj_len = rewards.shape[-1]

      last_gae_lam = 0
      for step in reversed(range(traj_len)):
          next_non_terminal = masks[:,step]
          next_values = values[:,step + 1]
          delta = rewards[:,step] + self.gamma * next_values * next_non_terminal - values[:,step]
          last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
          advantages[:,step] = last_gae_lam
      # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
      # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
      returns = advantages + values[:,:-1]
      return returns


  def update(self, exp, context, train=True):
    """Updates the posterior using linear bayesian regression formula."""

    # rewards = self.normalize_reward(rewards)
    self.model.eval()
    self.t += 1
    context = context.astype(np.float64)
    self.storage.save_exp(exp, context)

    fragment_max = len(exp.reward) // self.horizon
    while fragment_max >= self.fragments:
        self.add_bandit_fragment()
        self.fragments += 1


    if self.t % self.update_freq_nn == 0 and self.t >= self.batch_size and train:

      data = self.storage.get_data()
      mean_loss = self.train(data)
      self.model.eval()
      for i in range(self.fragments):
          self.precision[i] = self.lambda_prior * torch.eye(self.param_dim, device=self.device, dtype=self.dtype)
          self.f[i] = self.precision[i] @ self.model.get_last_layer_weights().to(self.dtype)#torch.zeros(self.param_dim, device=self.device, dtype=self.dtype)

      i = 0
      for j in range(len(data)):

          exp, context = data[j]
          step_idx = exp.step[:-1]
          with torch.no_grad():
              states = torch.tensor(np.array(exp.obs)).float()
              context = torch.tensor(context).float()
              context1 =  torch.stack([context for _ in range(len(states))])
              target_values, _ = self.target_model(states.to(self.device), context1.to(self.device))
              target_values = target_values[:, 0].unsqueeze(0)
              returns = self.compute_returns_and_advantage(torch.tensor(exp.mask).unsqueeze(0),
                                                           torch.tensor(exp.reward).unsqueeze(0),
                                                           target_values.cpu())  # returns is 1 shorter than H
              returns = returns[0].to(self.dtype)
              context1 = context1[:-1]
              states   = states[:-1]
              for i in range(self.fragments):
                  idx = i * self.horizon

                  #Sample from traj because there is corelation between in-traj samples
                  contexts = context1[idx:idx+1].to(self.device)
                  first_states = states[idx:idx+1].float().to(self.device)
                  values  =returns[idx:idx+1].to(self.device)


                  new_z = self.model.encode(first_states, contexts)
                  new_z = new_z.to(self.dtype)

                  # The algorithm could be improved with sequential formulas (cheaper)
                  self.precision[i] += torch.matmul(new_z.T, new_z)
                  self.f[i] += torch.matmul(values, new_z)

      for i in range(self.fragments):
          # self.cov[i] = torch.linalg.inv(self.precision[i])
          self.mu[i] = torch.linalg.solve(self.precision[i], self.f[i])

    else:
        mean_loss = 0
        with torch.no_grad():

            step_idx = exp.step[:-1]
            states = torch.tensor(np.array(exp.obs)).to(self.device).float()
            if len(states) == 1:
                ipdb.set_trace()
            context = torch.tensor(context).float()
            contexts = torch.stack([context for _ in range(len(states))])
            contexts = contexts.to(self.device)
            target_values, _ = self.target_model(states, contexts)
            target_values = target_values[:,0].unsqueeze(0)
            returns = self.compute_returns_and_advantage(torch.tensor(exp.mask).unsqueeze(0).to(self.device), torch.tensor(exp.reward).unsqueeze(0).to(self.device), target_values)
            returns = returns[0].to(self.device)
            states = states[:-1]
            contexts = contexts[:-1]

            for i in range(self.fragments):
                idx = i * self.horizon
                returns = returns[idx:idx+1]
                states = states[idx:idx+1]
                contexts = contexts[idx:idx+1]

                phi = self.model.encode(states, contexts)
                phi = phi.to(self.dtype)
                # Retrain the network on the original data (data_h)
                self.precision[i] += torch.matmul(phi.T, phi)
                self.f[i] += torch.matmul(returns, phi)
                # self.cov[i] = torch.linalg.inv(self.precision[i])
                self.mu[i] = torch.linalg.solve(self.precision[i], self.f[i])
                # self.mu[i] = torch.matmul(self.cov[i], self.f[i])

                # Inverse Gamma posterior update
                self.a[i] += 0.5
                # b_upd = 0.5 * (self.yy[i] - torch.matmul(self.mu[i], torch.matmul(self.precision[i], self.mu[i])))
                # self.b[i] = self.b0 + b_upd
    return mean_loss

  def train(self, data):
        self.model.train()
        dataset = NeuralLinearDataset(data, self.num_unroll_steps, self.stacked_observations, self.horizon)

        smplr = torch.utils.data.RandomSampler(dataset,
                                               replacement=True,
                                               num_samples=self.training_iter * self.batch_size)
        dataloader = DataLoader(dataset,
                           batch_size= self.batch_size,
                           sampler=smplr,
                           shuffle=False,
                           pin_memory=True,
                           num_workers=0,
                           collate_fn=my_collate)


        num_iter = 0
        mean_loss = 0
        for i, sample in enumerate(dataloader):
            self.training_steps += 1
            if self.training_steps % self.target_model_update == 0:
                self.soft_update_from_to(self.model, self.target_model)
                self.target_model.eval()
            loss = self.train_step(sample)
            num_iter += 1
            mean_loss += loss
            # print( f' Training step: {num_iter}/{self.training_iter}. loss={loss}',end="\r")

        return mean_loss / num_iter

  # def train_step(self, sample):
  #     context, obs, _, reward, mask= sample
  #     # context = torch.tile(context,[1,obs.shape[1],1])
  #     obs = obs.to(self.device)
  #     B,H,_ = obs.shape
  #     context = torch.stack([context for _ in range(H)],dim=1) #B x H x C
  #
  #     reward = reward.to(self.device)
  #     mask = mask.to(self.device)
  #     context = context.to(self.device)
  #
  #     obs1 = obs.reshape(B * H,obs.shape[-1])
  #     context1 = context.reshape(B * H,context.shape[-1])
  #
  #     with torch.no_grad():
  #         target_value, _  = self.target_model(obs1, context1)
  #         target_value = target_value.reshape(B, H)
  #         returns = self.compute_returns_and_advantage(mask, reward, target_value)
  #
  #     value_hat, phi = self.model(obs1, context1)
  #     value_hat = value_hat.reshape(B, H)
  #
  #
  #     loss = self.loss_fn(value_hat[:,:-1], returns)  # + self.loss_fn(context,z_hat)
  #     # loss += 0.1 * torch.mean(torch.norm(phi,dim=-1))
  #     total_loss = loss.item()
  #     self.optimizer.zero_grad()
  #     loss.backward()
  #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
  #     self.optimizer.step()
  #     self.scheduler.step()
  #
  #     return total_loss

  def train_step(self, sample):
      context, obs, _, reward, mask, steps = sample
      # context = torch.tile(context,[1,obs.shape[1],1])
      B = context.shape[0]

      loss = 0
      cnt = 0
      obs_arr = []
      return_arr = []
      for b in range(B):
          obs1 = obs[b].to(self.device)
          H = obs1.shape[0]
          if H == 1:
              return_arr.append(reward[b].to(self.device))
          else:

              cnt += 1
              context1 = torch.stack([context[b] for _ in range(H)])  #  H x C

              reward1 = reward[b].to(self.device)
              mask1 = mask[b].to(self.device)
              context1 = context1.to(self.device)


              with torch.no_grad():
                  target_value, _ = self.target_model(obs1, context1)
                  returns = self.compute_returns_and_advantage(mask1.unsqueeze(0), reward1.unsqueeze(0), target_value[:,0].unsqueeze(0))

              return_arr.append(returns[:,0])
          obs_arr.append(obs1[0])

      context = context.to(self.device)
      obs1 = torch.stack(obs_arr)
      returns = torch.cat(return_arr)
      value_hat, phi = self.model(obs1, context)
      loss += self.loss_fn(value_hat[:, 0], returns)  # + self.loss_fn(context,z_hat)
      loss += 0.1 * torch.mean(torch.norm(phi,dim=-1))

      loss = loss / cnt
      total_loss = loss.item()
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
      self.optimizer.step()
      self.scheduler.step()

      return total_loss

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior


class NeuralLinearDataset(Dataset):

  def __init__(self, data, num_unroll, stacked_observations, horizon):
    self.data = data
    self.num_unroll = num_unroll
    self.stacked_observations = stacked_observations
    self.horizon = horizon

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    exp = self.data[item][0]
    traj_len = exp.reward.shape[0]
    t = np.random.randint(traj_len)
    policy_vector =  self.data[item][1]
    policy_vector = torch.tensor(policy_vector).float()
    # t = 0
    # obss = [torch.tensor(str_to_arr(o)) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations-1]]
    obss = [torch.tensor(o) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations-1]]
    obss = torch.stack(obss,dim=0) # T x H x W x C
    actions = exp.action[t:t+self.num_unroll]
    steps = exp.step[t:t+self.num_unroll]
    rewards = exp.reward[t:t+self.num_unroll]
    # rewards = exp.reward
    masks = exp.mask[t:t+self.num_unroll]
    actions, rewards, masks, steps = torch.tensor(actions).float(), torch.tensor(rewards).float(), torch.tensor(masks).float(), torch.tensor(steps).float()
    # if t >= traj_len - self.num_unroll:
    #     obss_pad = torch.zeros([self.num_unroll - traj_len + t] + list(obss.shape[1:]))
    #     rewards_pad = torch.zeros([self.num_unroll - traj_len + t])
    #     actions_pad = torch.zeros([self.num_unroll - traj_len + t] + list(actions.shape[1:]))
    #     masks_pad = torch.zeros([self.num_unroll - traj_len + t])
    #     obss = torch.cat([obss, obss_pad],dim=0)
    #     rewards = torch.cat([rewards, rewards_pad], dim=0)
    #     actions = torch.cat([actions, actions_pad], dim=0)
    #     masks = torch.cat([masks, masks_pad], dim=0)

    obss = obss.float() #/ 255.0 #uint8 --> float32

    return policy_vector, obss, actions, rewards, masks, steps


# def my_collate(batch):
#   policy_vec = [item[0] for item in batch]
#   obs = torch.stack([item[1] for item in batch])
#   action = torch.stack([item[2] for item in batch])
#   reward = torch.stack([item[3] for item in batch])
#   mask = torch.stack([item[4] for item in batch])
#   return [policy_vec, obs, action, reward, mask]

def my_collate(batch):
  policy_vec = torch.stack([item[0] for item in batch])
  obs = [item[1] for item in batch]
  action = [item[2] for item in batch]
  reward = [item[3] for item in batch]
  mask = [item[4] for item in batch]
  step = [item[5] for item in batch]
  return [policy_vec, obs, action, reward, mask, step]