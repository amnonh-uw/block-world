import tensorflow as tf
import numpy as np
import os

class Dagger:
    def __init__(self, env, **kwargs):
        self.render = True
        self.num_rollouts = 25
        self.train_batch_size = 25
        self.train_iterations = 10000
        self.iterations = 50
        self.train_report_frequency = 1000
        self.max_steps = env.spec.timestep_limit
        self.num_actions = env.action_space.n
        self.__dict__.update(kwargs)

        self.env = env
        self.obs_shape = env.observation_space.shape
        self.act_shape = ()

        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

    def learn(self, policy_class, fname):
        self.build_graph(policy_class)
        self.expert_step()

        # record return and std for plotting
        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.policy.policy_initializer()

            for i in range(self.iterations):
                print("DAgger iteration {}".format(i))
                self.train_step()
                self.test_step()

            self.save_policy(fname)

        dagger_results = {'means': self.save_mean, 'stds': self.save_std, 'train_size': self.save_train_size,
                          'expert_mean': self.save_expert_mean, 'expert_std': self.save_expert_std}

        print("DAgger iterations finished!")
        print(dagger_results)

    def test(self, policy_class, fname):
        self.build_test_graph(policy_class)
        self.load_policy(fname)

    def build_graph(self, policy_class):
        self.x = tf.placeholder(tf.float32, name="x", shape=(None,) + self.obs_shape)
        self.y = tf.placeholder(tf.int32, name="y", shape=(None,) + self.act_shape)

        with tf.variable_scope("policy"):
            self.policy = policy_class(self.x, self.y, self.num_actions)
            self.yhat = self.policy.get_output()
            self.loss = self.policy.get_loss()
            self.train_step_op = tf.train.AdamOptimizer().minimize(self.loss)

    def build_test_graph(self, policy_class):
        self.x = tf.placeholder(tf.float32, shape=(None,) + self.obs_shape)
        with tf.variable_scope("policy"):
            self.policy = policy_class(self.x, None, self.num_actions)
            self.yhat  = self.policy.get_output()

    def save_policy(self, fname):
        fname += "/"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy"))
        saver.save(tf.get_default_session(), fname)

    def load_policy(self, fname):
        fname += "/"
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), fname)

    def eval_policy(self, obs):
        return self.yhat.eval(feed_dict={self.x: obs[None, :]})

    def expert_step(self):
        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
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

        self.save_expert_mean = np.mean(returns)
        self.save_expert_std = np.std(returns)

        print("expert returns", returns)
        print("expert mean return", self.save_expert_mean)
        print("expert std of return", self.save_expert_std)

        self.obs_data = np.squeeze(np.array(observations))
        self.act_data = np.squeeze(np.array(actions))

    def train_step(self):
        # do we need to reset adam at this point?

        for step in range(self.train_iterations):
            batch_i = np.random.randint(0, self.obs_data.shape[0], size=self.train_batch_size)
            self.train_step_op.run(feed_dict={self.x: self.obs_data[batch_i,], self.y: self.act_data[batch_i,]})
            if (step % self.train_report_frequency == 0):
                loss = self.loss.eval(feed_dict={self.x: self.obs_data, self.y: self.act_data})
                print ("train step {} objective loss {}".format(step, loss))

    def test_step(self):
        returns = []
        observations = []
        actions = []
        expert_actions = []

        for i in range(self.num_rollouts):
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