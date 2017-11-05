import gym
import numpy as np
from block_world_core import env as make_env
import tensorflow as tf

class BlockWorldEnv(gym.Env):
    class ActionSpace(gym.Space):
        #
        # Actions are a vector of [-1,1],[-1,1],[-1,1]. Each colummn represents a movement in the x,y or z direction
        # all zeros mean episode end
        #
        def __init__(self):
            self.n = 32
            self.dtype = tf.int32
            self.shape = 1

    class EnvSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.metadata = {'render.modes': ['human', '']}
        self.action_space = self.ActionSpace()
        self._spec = self.EnvSpec(timestep_limit=35)

        self._seed()
        self.reset_counter = 0
        self.step_counter = 0

        kwargs.pop("show_obs", None)
        kwargs.pop("reach_minimum", None)
        kwargs.pop("step_size", None)
        kwargs.pop("run", None)
        kwargs.pop("verbose", None)

        self.block_env = make_env(run=self.run, verbose=True, params_args=kwargs)

    def obs(self):
        return self.block_env.obs_dict()

    def expert_action(self):
        dist =  self.block_env.target_pos - self.block_env.finger_pos
        return dist

    def target_reached(self):
        dist = abs(self.block_env.finger_pos - self.block_env.target_pos)
        return (dist > self.reach_minimum).sum() == 0

    def _step(self, action):
        if abs(action[0]) < self.reach_minimum and \
           abs(action[1]) < self.reach_minimum and \
           abs(action[2]) < self.reach_minimum:
                self.episode_ended = True
                if self.target_reached():
                    # Halelujah
                    reward = 1000
                    self.episode_ended = True
                else:
                    reward = -1000
        else:
            inc_pos = action
            self.block_env.move_finger(inc_pos)
            if self.block_env.collision:
                self.episode_ended = True
                reward = -500
            else:
                reward = -1

        if self.step_counter > self.max_steps:
            self.episode_ended = True

        reward = float(reward) / 1000.0

        return self.obs(), reward, self.episode_ended, None

    def _reset(self):
        self.reset_counter += 1
        self.max_steps = 30
        self.step_counter = 0
        self.episode_ended = False

        self.block_env.reset()

        return self.obs()

    def _render(self, mode='human', close=False):
        if close:
            return

        if mode is 'human' or mode is '':
            print("target: " + str(self.block_env.target_pos) + " finger: " + str(self.block_env.finger_pos))
        else:
            super(gym.env, self).render(mode=mode)  # just raise an exception

    def save_cams(self, path):
        self.block_env.save_cams(path)

    def save_positions(self, path):
        self.block_env.save_positions(path)

    def _seed(self, seed=None):
        return []

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass
