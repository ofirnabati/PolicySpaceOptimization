# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a family of neural network architectures for bandits.

The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from muzero_models import DownSample
# import math
import ipdb


class NormalizationLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n = x.norm(dim=-1,keepdim=True)
        n = n.detach()
        x = x / n
        return x

def init_params_gauss(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class NeuralBanditModel(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModel, self).__init__()

        self.hparams = hparams
        self.fragments = hparams.fragments

        policy_layers = []
        policy_layers.append(nn.Linear(hparams.context_dim, hparams.layers_size[0] // 2 ,bias=False))
        policy_layers.append(nn.BatchNorm1d( hparams.layers_size[0] // 2))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0] * 2, hparams.layers_size[0] ))
        # policy_layers.append(nn.BatchNorm1d(hparams.layers_size[0]))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0] ))
        self.policy_embedder = nn.Sequential(*policy_layers)
        # if hparams.env == 'breakout':
        #     self.block_output_size_policy = (
        #                 16
        #                 * math.ceil(hparams.obs_shape[1] / 16)
        #                 * math.ceil(hparams.obs_shape[2] / 16)
        #         )
        #
        #     self.downsampler = DownSample(hparams.obs_shape[0], 64)
        #     self.conv1x1 = nn.Conv2d(64, 16, 1)
        #     self.bn_state =  nn.BatchNorm2d(16)
        #     self.state_fc = nn.Linear(self.block_output_size_policy, hparams.layers_size[0] // 2)
        #     self.state_embedder = nn.Sequential(self.downsampler, self.conv1x1, self.bn_state, nn.ReLU())
        # else:
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Linear(hparams.fragments, hparams.layers_size[0] // 2, bias=False)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0])
        layers = []

        for i in range(len(hparams.layers_size)-1):
            layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1]))
            if i < len(hparams.layers_size) - 2:
                layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                layers.append(nn.ReLU())

            # if i<len(hparams.layers_size)-2:
            #     layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
            #     layers.append(nn.ReLU())
            # else:
            #     layers.append(nn.Tanh())
        # layers.append(NormalizationLayer())
        self.feature_extractor = nn.Sequential(*layers)
        self.value_pred = nn.Linear(hparams.layers_size[-1], 1, bias=False)
        self.feat_dim = hparams.layers_size[-1]
        # Initialize parameters correctly
        # self.apply(init_params)

        # decoder_layers = []
        # for j in range(len(hparams.layers_size),1,-1):
        #         decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
        #         decoder_layers.append(nn.ReLU())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        # self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)


    def forward(self, state, policy):
        phi   = self.encode(state, policy)
        # phi   = self.encode(policy)
        value = self.value_pred(phi)
        # phi_hat = self.decoder(phi)
        return value, phi

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        state = F.one_hot(state, self.fragments)
        state = state.float()
        s = self.state_embedder(state)
        # if self.hparams.env == 'breakout':
        #     s = s.view(-1,self.block_output_size_policy)
        #     s = self.state_fc(s)
        z = self.policy_embedder(policy)
        x = torch.cat([s,z],dim=-1)
        # x = z
        # x = self.bn0(z)
        x = F.relu(x)
        # return z
        # x = z
        return self.feature_extractor(x)
        # return policy

    # def decode(self, phi):
    #     z_hat = self.decoder(phi)
    #     return z_hat

    # def estimate_advantages(self, rewards, masks, values):
    #     """
    #
    #     :param rewards:   Time x 1
    #     :param masks:  Time x 1
    #     :param values:  Time x 1
    #     :return:  Time x 1
    #     """
    #     deltas = torch.zeros_like(rewards)
    #     advantages = torch.zeros_like(rewards)
    #
    #     prev_value = 0 #torch.zeros(B, device= self.device)
    #     prev_advantage = 0 #torch.zeros(B, device= self.device)
    #     for i in reversed(range(rewards.size(0))):
    #         deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
    #         advantages[i] = deltas[i] + self.gamma * self.gae_lambda * prev_advantage * masks[i]
    #
    #         prev_value = values[i]
    #         prev_advantage = advantages[i]
    #
    #     returns = values + advantages # Time x 1
    #     # advantages = (advantages - advantages.mean()) / advantages.std()
    #
    #     return returns

    def get_last_layer_weights(self):
        return self.value_pred.weight.data[0]


