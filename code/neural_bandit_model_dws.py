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
from dwsnet.models import DWSModel
from dwsnet.layers import BN, DownSampleDWSLayer, Dropout, DWSLayer, InvariantLayer, ReLU

def google_nonlinear(x):
    return torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) + 0.001 * x


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


class NeuralBanditModelVAEGaussian(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModelVAEGaussian, self).__init__()
        self.state_based = hparams.state_based_value
        self.no_embedding = hparams.no_embedding
        # self.non_linear_func = nn.ReLU
        if self.state_based:
            latent_dim = hparams.layers_size[0] // 2
        else:
            latent_dim = hparams.layers_size[0]
        self.hparams = hparams
        self.policy_embedder = DWSModel(
            weight_shapes=hparams.weight_shapes,
            bias_shapes=hparams.bias_shapes,
            input_features=1,
            hidden_dim=hparams.dim_hidden,
            n_hidden=hparams.n_hidden,
            reduction=hparams.reduction,
            n_fc_layers=hparams.n_fc_layers,
            set_layer=hparams.set_layer,
            dropout_rate=hparams.do_rate,
            bn=hparams.add_bn)


        state_layers = []
        state_layers.append(nn.Linear(hparams.obs_dim, latent_dim, bias=True))
        # state_layers.append(nn.BatchNorm1d( latent_dim))
        state_layers.append(self.non_linear_func())
        state_layers.append(nn.Linear(latent_dim, latent_dim, bias=True))
        state_layers.append(self.non_linear_func())
        # state_layers.append(nn.BatchNorm1d(latent_dim))
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Sequential(*state_layers)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0] // 2)


        # self.fc_mu = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])
        # self.fc_var = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])

        self.relu = ReLU()
        self.clf = InvariantLayer(
            weight_shapes=hparams.weight_shapes,
            bias_shapes=hparams.bias_shapes,
            in_features=hparams.hidden_dim,
            out_features=1,
            reduction= hparams.reduction,
            n_fc_layers= hparams.n_out_fc,
            bias=False
        )

        # self.value_pred = nn.Linear(hparams.layers_size[-1], 1, bias=False)
        self.feat_dim = self.clf.latent_dim
        # Initialize parameters correctly
        # self.apply(init_params)

        # decoder_layers = []
        # for j in range(len(hparams.layers_size),1,-1):
        #         decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
        #         # decoder_layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
        #         decoder_layers.append(self.non_linear_func())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        # decoder_layers.append(self.non_linear_func())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        # decoder_layers.append(self.non_linear_func())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        # self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)


    def forward(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy
        else:
            # mu, _   = self.encode(state, policy)
            # mean_value = self.value_pred(mu)
            # mean_value = google_nonlinear(mean_value)
            x = self.policy_embedder(policy)
            x = self.relu(x)
            mean_value, mu = self.clf(x)
            return mean_value, mu

    def forward_sample(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy, None, None
        else:
            x = self.policy_embedder(policy)
            x = self.relu(x)
            mean_value, mu = self.clf(x)
            return mean_value, mu, None, None

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.no_embedding:
            return policy, None
        else:
            x = self.policy_embedder(policy)
            x = self.relu(x)
            mu = self.clf.extract_latent(x)
            return mu, None

    # def decode(self, z):
    #     return self.decoder(z)

    def sample(self, state, policy):
        if self.no_embedding:
            return policy
        else:
            x = self.policy_embedder(policy)
            x = self.relu(x)
            mu = self.clf.extract_latent(x)
            return mu


    def get_last_layer_weights(self):
        return self.clf.last_layer.weight.data[0]