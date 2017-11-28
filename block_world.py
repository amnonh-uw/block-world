import gym
import numpy as np
from block_world_core import env as make_env
import tensorflow as tf

class BlockWorldEnv(gym.Env):
    class EnvSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.metadata = {'render.modes': ['human', '']}
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
        self.action = None
        self.probe_direction = None

    def obs(self):
        obs_dict = self.block_env.obs_dict()
        if self.action is not None:
            obs_dict['action'] = self.action
        if self.probe_direction is not None:
            obs_dict['probe_direction'] = self.probe_direction

        return obs_dict

    def expert_action(self):
        dist =  self.block_env.target_pos - self.block_env.finger_pos
        self.action = dist
        return dist

    def target_reached(self):
        dist = abs(self.block_env.finger_pos - self.block_env.target_pos)
        return (dist > self.reach_minimum).sum() == 0

    def random_probe(self):
        self.probe_direction = np.random.random_sample(size=3) - 0.5
        self.block_env.probe_finger(self.probe_direction)
        return self.obs()

    def _step(self, action):
        def step_range(x):
            if x > self.step_size:
                return self.step_size
            if x < -self.step_size:
                return -self.step_size

            return x

        def not_visible(screen_pos):
            width, height = self.block_env.centercam.size
            if screen_pos[0] < 0.0 or screen_pos[0] >= width:
                return True
            if screen_pos[1] < 0.0 or screen_pos[0] >= height:
                return True
            return False

        if (abs(action) < self.reach_minimum).sum() == 3:
            self.episode_ended = True
            if self.target_reached():
                # Halelujah
                reward = 1000
                self.episode_ended = True
            else:
                reward = -1000
        else:
            # limit step sizes to range
            inc_pos = np.array([step_range(x)  for x in action])
            self.block_env.move_finger(inc_pos)
            if self.block_env.collision:
                self.episode_ended = True
                reward = -500
            elif not_visible(self.block_env.target_screen_pos):
                reward = -1000
                self.episode_ended = True
            elif not_visible(self.block_env.finger_screen_pos):
                reward = -1000
                self.episode_ended = True
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
