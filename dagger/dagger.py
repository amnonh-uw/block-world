import tensorflow as tf
import numpy as np
import os
import h5py

class Dagger:
    def __init__(self, env, policy_class, **kwargs):
        self.render = True
        self.num_rollouts = 25
        self.train_batch_size = 25
        self.train_epochs = 10
        self.iterations = 50
        self.train_report_frequency = 1000
        self.max_steps = env.spec.timestep_limit
        self.num_actions = env.action_space.n
        self.dir_name = None

        self.__dict__.update(kwargs)
        self.env = env
        self.policy_class = policy_class
        self.obs_shape = env.observation_space.shape
        self.act_shape = ()

        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

        self.samples = []

    def add_sample(self, observation, action, phase=None, rollout=None, step=None):
        sample = self.policy.save(observation, action, phase, rollout, step)
        self.samples.append(sample)
        return sample

    def learn(self, save_file_name):
        self.build_graph(self.policy_class)
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
                self.test_step(iter=i+1)

            self.save_policy(save_file_name)

        dagger_results = {'means': self.save_mean, 'stds': self.save_std, 'train_size': self.save_train_size,
                          'expert_mean': self.save_expert_mean, 'expert_std': self.save_expert_std}

        print("DAgger iterations finished!")
        print(dagger_results)

    def test(self, fname):
        self.build_test_graph(self.policy_class)
        self.load_policy(fname)

    def build_graph(self, policy_class):
        self.x = tf.placeholder(tf.float32, name="x", shape=(None,) + self.obs_shape)
        self.y = tf.placeholder(tf.int32, name="y", shape=(None,) + self.act_shape)

        with tf.variable_scope("policy"):
            self.policy = policy_class(self.x, self.y, self.num_actions, self.dir_name)
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

        for i in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = self.env.expert_action()

                # data aggregation
                path = self.add_sample(obs, action, phase=0, rollout=i, step=steps)
                self.env.save_cams(path)
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

    def train_step(self):
        sess = tf.get_default_session()
        # do we need to reset adam at this point?

        dataset = self.policy.get_dataset(self.samples)
        dataset = dataset.shuffle(len(self.samples))
        dataset = dataset.repeat(self.train_epochs)
        dataset = dataset.batch(self.train_batch_size)

        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()

        step = 0
        while True:
            try:
                obs_batch, action_batch = sess.run(get_next)
                _, loss = sess.run([self.train_step_op, self.loss], feed_dict={self.x: obs_batch, self.y: action_batch})
                step += 1
                if (step % self.train_report_frequency == 0):
                    print ("train step {} objective batch loss {}".format(step, loss))
            except tf.errors.OutOfRangeError:
                break

    def test_step(self, iter=None):
        returns = []

        train_size = len(self.samples)

        for i in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = self.yhat.eval(feed_dict={self.x: obs[None, :]})
                expert_action = self.env.expert_action()

                # data aggregation
                path = self.add_sample(obs, expert_action, phase=iter, rollout=i, step=steps)
                self.env.save_cams(path)
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


        # record mean return & std
        self.save_mean = np.append(self.save_mean, np.mean(returns))
        self.save_std = np.append(self.save_std, np.std(returns))
        self.save_train_size = np.append(self.save_train_size, train_size)