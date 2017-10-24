import gym
from gym import error
from gym.utils import seeding
from multi_continous import MultiContinuous
from multi_discrete import MultiDiscrete
from multi_discrete_and_gaussian import MultiDiscreteAndGaussian
import numpy as np
import math

class DistworldEnv(gym.Env):
    def __init__(self,
                 span=10,
                 dims = 3,
                 single_dim_action = False,
                 greedy=False,
                 column_greedy=False,
                 l2_penalty=None,
                 continous_actions=None,
                 reach_minimum = 0.1,
                 max_far=None,
                 no_stops = False):
        self.metadata = {'render.modes': ['human', '']}

        if greedy and column_greedy:
            raise ValueError('either greedy or column greedy but not both')

        if single_dim_action:
            if continous_actions:
                #
                # Discrete scalar to choose stop or select dimension to move in
                # Continous scalar for movement value
                #
                self.action_space = MultiDiscreteAndGaussian([0,dims], 1)
            else:
                #
                # Discrete vector up, down or nop for each dimension
                #
                self.action_space = MultiDiscrete([[0, dims * 2]])
        else:
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
        self.single_dim_action = single_dim_action
        self.greedy = greedy
        self.column_greedy = column_greedy
        self.max_far = max_far
        self.continous_actions = continous_actions
        self.reach_minimum = reach_minimum
        self.l2_penalty = l2_penalty
        self.no_stops = no_stops
        self._seed()

        self.reset_counter = 0
        self.step_counter = 0

    def map_discrete_action(self, n):
        if self.single_dim_action:
            n -= 1              # zero would have meant episode end
            act_dim = n // 2
            n = n % 2
            a = np.zeros([self.dims])
            if n == 0:
                a[act_dim] = 1
            else:
                a[act_dim] = -1
        else:
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
            a = np.array(a)

        return a

    def expert_action(self):
        dist = abs(self.finger_pos - self.target_pos)
        n = 0
        for i in range(self.dims):
            n *= 3
            if abs(dist[i] >= self.reach_minimum):
                if self.single_dim_action:
                    if dist[i] > 0:
                        return 1 + i * 2
                    else:
                        return 2 + i * 2
                else:
                    if dist[i] < 0:
                        n += 1
                    else:
                        n += 2
        return n

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
            if not self.no_stops:
                self.episode_ended = True
            return -1000

        old_dist = abs(old_pos - self.target_pos)
        new_dist = abs(self.finger_pos - self.target_pos)

        if self.greedy:
            if (old_dist < new_dist).sum() != 0:
                # bad move. We don't want to see regression in any dimension
                return -100

        if self.column_greedy:
            num_regressed_dims = (old_dist < new_dist).sum()
            if num_regressed_dims != 0:
                # bad move, regressed in num_regressed_dims dimensions
                return -100 * num_regressed_dims

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

                if not self.no_stops:
                    self.episode_ended = True
        else:
            if self.continous_actions:
                inc_pos = self.action_space.to_continous(action)
            else:
                inc_pos = self.map_discrete_action(discrete_action)

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
            if self.max_far is not None:
                low = np.maximum(self.finger_pos - self.max_far, -self.span)
                high = np.minimum(self.finger_pos + self.max_far, +self.span)
            else:
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
        if mode is 'human' or mode is '':
            # print(mode + ":target " + str(self.target_pos) + " finger " + str(self.finger_pos))
            pass
        else:
            super(DistworldEnv, self).render(mode=mode)  # just raise an exception

    def _seed(self, seed=None): return []

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass
