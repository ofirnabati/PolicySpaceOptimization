from gym.envs.mujoco import AntEnv
import numpy as np

class SparseAntV1(AntEnv):
    """Sparse Half-cheetah environment with target direction
    """
    def __init__(self, sparse_dist=7.0):
        self._goal_dir = 1.0
        self._sparse_dist = sparse_dist
        super().__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel * (np.abs(xposafter) >= self._sparse_dist)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,reward_ctrl=-ctrl_cost , sparse_dist = self._sparse_dist, xposafter=xposafter)
        return (observation, reward, done, infos)
