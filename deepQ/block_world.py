import gym
from gym import error
from gym.utils import seeding
from multi_continous import MultiContinuous
from multi_discrete import MultiDiscrete
from  block_world_core import env
import numpy as np

"""
    When implementing an environment, override the following methods
    in your subclass:

        _step
        _reset
        _render
        _close
        _configure
        _seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
"""

class BlockWorldEnv(gym.Env):
    def __init__(self, pos_unit = None, rot_unit = None):
        self.pos_unit = pos_unit
        self.rot_unit = rot_unit
        self.env = env(pos_unit=pos_unit, rot_unit=rot_unit)
        self.action_space = MultiDiscrete([[-1,1], [-1,1], [0,1]]) # no point in moving backwords
        self.observation_space = MultiContinuous(12)
        self._seed()

        self.reset_counter = 0
        self.step_counter = 0

    def obs(self):
        obs = self.env.finger_pos
        obs = np.append(obs, self.env.finger_rot)
        obs = np.append(obs, self.env.target_pos)
        obs = np.append(obs, self.env.target_rot)

        return obs

    def calc_reward(self):
        r = 0
        if self.env.collision:
            self.episode_ended = True
            r = -1000
        else:
            d = np.linalg.norm(self.env.finger_pos - self.env.target_pos)
            if d < self.best_distance:
                print("improved distance from " + str(self.best_distance) + " to " + str(d))
                r = int(100 * (self.best_distance - d))
                self.best_distance = d

        return r

    def _step(self, action):
        self.step_counter += 1

        # returns new_obs, rew, done, _

        def map_action(n):
            if n == 0:
                return 0
            if n == 1:
                return self.pos_unit
            if n == 2:
                return -self.pos_unit

            print("map action impossible " + str(n))

        def target_reached():
            for i in range(3):
                if abs(self.env.finger_pos[i] - self.env.target_pos[i]) > self.pos_unit:
                    return False

            return True

        if action == 0:

            self.episode_ended = True
            if target_reached():
                reward = 1000 - self.step_counter        # large reward
            else:
                reward = 0
                print("action: end episode reward " + str(reward))
        else:
            x = map_action(action % 3)
            action = action // 3
            y = map_action(action % 3)
            action = action // 3
            z = map_action(action)
            assert (z >= 0)     # no point in moving backwords

            inc_pose = str(x) + "," + str(y) + "," + str(z)

            self.env.move_finger(inc_pose)
            reward = self.calc_reward()
            print(str(self.step_counter) + " move finger " + str(inc_pose) + " reward " + str(reward))

        if self.step_counter > self.max_steps:
            self.episode_ended = True


        return self.obs(), reward, self.episode_ended, None

    def _reset(self):
        self.reset_counter += 1
        self.max_steps = 30
        self.step_counter = 0
        print("reset: " + str(self.reset_counter))
        self.env.reset(tray_length=3.0, tray_width=2.0, stereo_distance=0.2, max_objects=0)
        self.episode_distance = np.linalg.norm(self.env.finger_pos - self.env.target_pos)
        print("Episode distance " + str(self.episode_distance))
        self.best_distance = self.episode_distance
        self.episode_ended = False

        return self.obs()

    def _render(self, mode='human', close=False):
        if close:
            return
        raise NotImplementedError

    def _seed(self, seed=None): return []

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass
