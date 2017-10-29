import gym
from gym import error
from gym.utils import seeding
from multi_continous import MultiContinuous
from multi_discrete import MultiDiscrete
from multi_discrete_and_gaussian import MultiDiscreteAndGaussian
import numpy as np
import math

class envspec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DistworldEnv(gym.Env):
    def __init__(self,
                 span=10,
                 dims = 3,
                 greedy=False,
                 l2_penalty=None,
                 continous_actions=None,
                 reach_minimum = 0.1):

        self._spec = envspec(timestep_limit=30)
        self.metadata = {'render.modes': ['human', '']}

        if continous_actions:
            #
            # Boolean to choose stop or continue to move
            # Continous vector for movement value
            #
            self.action_space = MultiDiscreteAndGaussian([0, 1], dims)
        else:
            #
            # Discrete scalar for up, down, or nop for each dimensoin
            #
            act = list()
            for _ in range(dims): act += [[-1, 1]]
            self.action_space = MultiDiscrete(act)

        self.observation_space = MultiContinuous(2 * dims)
        self.span = span
        self.dims = dims
        self.greedy = greedy
        self.continous_actions = continous_actions
        self.reach_minimum = reach_minimum
        self.l2_penalty = l2_penalty
        self._seed()

        self.reset_counter = 0
        self.step_counter = 0

    def map_discrete_action(self, n):
        a = list()
        for _ in range(self.dims):
            k = n % 3
            n = n // 3
            if k == 0:
                a += [0]
            if k == 1:
                a += [1]
            if k == 2:
                a += [-1]

        return np.array(a)

    def expert_action(self):
        dist = self.target_pos - self.finger_pos
        act = 0
        n = 1

        for i in range(self.dims):
            if abs(dist[i]) >= self.reach_minimum:
                if dist[i] > 0:
                    act += 1 * n
                else:
                    act += 2 * n
            n *= 3

        return act

    def target_reached(self):
        dist = abs(self.finger_pos - self.target_pos)
        return (dist > self.reach_minimum).sum() == 0

    def obs(self):
        obs = self.finger_pos
        obs = np.append(obs, self.target_pos)

        return obs

    def calc_intermediate_reward(self, old_pos):
        if (abs(self.finger_pos) > self.span).sum() != 0:
            # stepped outside of allowed ranges
            self.episode_ended = True
            return -1000

        old_dist = abs(old_pos - self.target_pos)
        new_dist = abs(self.finger_pos - self.target_pos)

        if self.greedy:
            if (old_dist < new_dist).sum() != 0:
                # bad move. We don't want to see regression in any dimension
                return -100

        # Every steps costs
        return -1

    def _step(self, action):
        self.step_counter += 1

        discrete_action = self.action_space.to_discrete(action)
        if discrete_action == 0:
            if self.target_reached():
                # Halelujah
                reward = 1000
                self.episode_ended = True
            else:
                # Moving more would have been better
                if self.l2_penalty is not None:
                    reward = -math.ceil(self.l2_penalty * np.linalg.norm(self.target_pos - self.finger_pos))
                else:
                    reward = -1000

                self.episode_ended = True
        else:
            if self.continous_actions:
                inc_pos = self.action_space.to_continous(action)
            else:
                inc_pos = self.map_discrete_action(discrete_action)

            # print("step {}".format(inc_pos))

            old_pos = self.finger_pos
            self.finger_pos = self.finger_pos + inc_pos
            reward = self.calc_intermediate_reward(old_pos)

        if self.step_counter > self.max_steps:
            self.episode_ended = True

        reward = float(reward) / 1000.0

        # print(str(self.step_counter) + " action " + str(action) + ": reward " + str(reward) +
        #      " target " + str(self.target_pos) + " finger " + str(self.finger_pos))
        return self.obs(), reward, self.episode_ended, None

    def _reset(self):
        self.reset_counter += 1
        self.max_steps = 30
        self.step_counter = 0
        self.episode_ended = False

        def target_range():
            low = -self.span
            high = +self.span

            return low, high

        def randint(low, high):
            r = np.zeros(self.dims)
            if type(low) == np.ndarray:
                for i in range(self.dims):
                    r[i] = np.random.randint(low[i], high[i])
            else:
                for i in range(self.dims):
                    r[i] = np.random.randint(low, high)

            return r

        if self.continous_actions:
            self.finger_pos = np.random.uniform(-self.span, +self.span, self.dims)
            low, high = target_range()
            self.target_pos = np.random.uniform(low, high, self.dims)
        else:
            self.finger_pos = randint(-self.span, +self.span)
            low, high = target_range()
            self.target_pos = randint(low, high)

        # print("reset: " + str(self.reset_counter) + " finger  " + str(self.finger_pos) + " target " + str(self.target_pos))
        return self.obs()

    def _render(self, mode='human', close=False):
        # print('finger {} target {}'.format(self.finger_pos, self.target_pos))
        # print("render ####")
        pass

    def _seed(self, seed=None): return []

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass
