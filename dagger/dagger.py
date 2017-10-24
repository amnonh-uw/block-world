import tensorflow as tf
import numpy as np

# def policy(x, act_shape):
#     act_dim = act_shape[0]
#     h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
#     h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
#     h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
#     yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
#
#     return yhat

class Dagger:
    def __init__(self, env, **kwargs):
        self.render = True
        self.num_rollouts = 25
        self.train_batch_size = 25
        self.train_iterations = 10000
        self.iterations = 50
        self.train_report_frequency = 1000
        self.max_steps = env.spec.timestep_limit
        self.__dict__.update(kwargs)

        self.env = env

        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

    def learn(self, policy_network):
        self.expert_step()
        self.build_graph(policy_network)

        # record return and std for plotting
        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.iterations):
                print("DAgger iteration ".format(i))
                self.train_step
                self.test_step()

        dagger_results = {'means': self.save_mean, 'stds': self.save_std, 'train_size': self.save_train_size,
                          'expert_mean': self.save_expert_mean, 'expert_std': self.save_expert_std}

        print("DAgger iterations finished!")
        print(dagger_results)

    def build_graph(self, policy_network):
        obs_shape = self.obs_data.shape[1:]
        act_shape = self.act_data.shape[1:]

        self.x = tf.placeholder(tf.float32, shape=(None,) + obs_shape)
        self.y = tf.placeholder(tf.float32, shape=(None,) + act_shape)
        self.yhat = policy_network(self.x, act_shape)

        self.loss_l2 = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss_l2)

    def expert_step(self):
        print("expert step")

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            print("iter {}".format(i))
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = self.env.expert_action()
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.render:
                    self.env.render()

                if steps >= self.max_steps:
                    break

            returns.append(totalr)

        self.expert_mean = np.mean(returns)
        self.expert_std = np.std(returns)

        print("returns", returns)
        print("mean return", self.expert_mean)
        print("std of return", self.expert_std)

        self.obs_data = np.squeeze(np.array(observations))
        self.act_data = np.squeeze(np.array(actions))

    def train_step(self):
        # do we need to reset adam at this point?

        for step in range(self.train_iterations):
            batch_i = np.random.randint(0, self.obs_data.shape[0], size=self.train_batch_size)
            self.train_step.run(feed_dict={self.x: self.obs_data[batch_i,], self.y: self.act_data[batch_i,]})
            if (step % self.train_report_frequency == 0):
                print ("learn step {}".format(step))
                print ("objective loss", self.loss_l2.eval(feed_dict={self.x: self.obs_data, self.y: self.act_data}))

    def test_step(self):
        print("test step")

        returns = []
        observations = []
        actions = []
        expert_actions = []

        for i in range(self.num_rollouts):
            print("iter".format(i))
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = self.yhat.eval(feed_dict={self.x: obs[None, :]})
                expert_action = self.env.expert_action()
                observations.append(obs)
                actions.append(action)
                expert_actions.append(expert_action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.render:
                    self.env.render()

                if steps >= self.max_steps:
                        break
            returns.append(totalr)

            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            train_size = self.obs_data.shape[0]
            # data aggregation
            self.obs_data = np.concatenate((self.obs_data, np.array(observations)), axis=0)
            self.act_data = np.concatenate((self.act_data, np.array(expert_actions)), axis=0)

            # record mean return & std
            self.save_mean = np.append(self.save_mean, np.mean(returns))
            self.save_std = np.append(self.save_std, np.std(returns))
            self.save_train_size = np.append(self.save_train_size, train_size)