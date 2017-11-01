import gym
import numpy as np
from block_world_core import env as make_env
import tensorflow as tf

class BlockWorldEnv(gym.Env):
    class ObsSpace(gym.Space):
        def __init__(self, width, height):
            # center cam + depth cam
            self.shape = tuple([width, height, 4])
            self.dtype = tf.float32

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
        self.observation_space = self.ObsSpace(self.width, self.height)
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
        c = np.array(self.block_env.centercam)
        c = c[:,:,:-1]
        d = np.array(self.block_env.multichanneldepthcam)
        d = np.expand_dims(d, axis=2)
        obs = np.concatenate((c,d), axis=2).astype(np.float32)

        print("obseravtion shape {}".format(obs.shape))
        return obs


    def target_reached(self):
        dist = abs(self.block_env.finger_pos - self.block_env.target_pos)
        return (dist > self.reach_minimum).sum() == 0

    def expert_action(self):
        dist = self.block_env.target_pos - self.block_env.finger_pos
        act = 0
        n = 1
        for i in range(3):
            if abs(dist[i]) >= self.reach_minimum:
                if dist[i] > 0:
                    act += 1 * n
                else:
                    act += 2 * n
            n *= 3

        return act

    def map_discrete_action(self, n):
        a = list()
        for _ in range(3):
            k = n % 3
            n = n // 3
            if k == 0:
                a += [0]
            if k == 1:
                a += [1]
            if k == 2:
                a += [-1]

        return np.array(a)

    def _step(self, action):
        discrete_action = action
        if discrete_action == 0:
            self.episode_ended = True
            if self.target_reached():
                # Halelujah
                reward = 1000
                self.episode_ended = True
            else:
                reward = -1000
        else:
            inc_pos = self.map_discrete_action(discrete_action) * self.step_size
            self.block_env.move_finger(inc_pos)
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
        print("_render_ ###################")
        if mode is 'human' or mode is '':
            # print(mode + ":target " + str(self.target_pos) + " finger " + str(self.finger_pos))
            pass
        else:
            super(BlockworldEnv, self).render(mode=mode)  # just raise an exception

    def save_cams(self, path):
        self.block_env.save_cams(path)

    def _seed(self, seed=None):
        return []

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass