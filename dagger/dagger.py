import tensorflow as tf
import numpy as np
import os
import h5py

class Dagger:
    def __init__(self, env, policy_class, **kwargs):
        self.render = True
        self.num_rollouts = 25
        self.batch_size = 25
        self.epochs = 10
        self.iterations = 50
        self.report_frequency = 1000
        self.save_frequency = 5000
        if env is not None:
            self.max_steps = env.spec.timestep_limit
        self.dir_name = None
        self.max_samples = None

        self.__dict__.update(kwargs)
        self.env = env
        self.policy_class = policy_class

        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

        self.samples = []

    def add_sample(self,  obs, action, phase=None, rollout=None, step=None):
        obs['action'] = action
        sample = self.policy.save_sample(obs, phase, rollout, step)
        if sample != None:
            self.samples.append(sample)
        return sample

    def learn_all_samples(self, save_file_name = None, load_file_name = None):
        if save_file_name is not None:
            self.save_file_name = save_file_name
        self.build_graph(self.policy_class)
        samples = tf.train.match_filenames_once(self.dir_name + '/*.tfrecord')
        if self.max_samples is not None:
            samples = samples[0:self.max_steps]
        dataset = self.policy.get_dataset(samples)
        dataset = dataset.shuffle(tf.size(samples, out_type=tf.int64))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if load_file_name == None:
                self.policy.policy_initializer()
            else:
                self.load_policy(load_file_name)

            self.train_step(dataset)
            self.save_policy(self.save_file_name)

    def learn(self, load_file_name = None):
        self.build_graph(self.policy_class)
        self.expert_step()

        # record return and std for plotting
        self.save_mean = []
        self.save_std = []
        self.save_train_size = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if load_file_name == None:
                self.policy.policy_initializer()
            else:
                self.load_policy(load_file_name)

            for i in range(self.iterations):
                print("DAgger iteration {}".format(i))
                self.train_step()
                self.save_policy(self.save_file_name)
                self.load_policy(self.save_file_name)
                self.test_step(iter=i+1)

        dagger_results = {'means': self.save_mean, 'stds': self.save_std, 'train_size': self.save_train_size,
                          'expert_mean': self.save_expert_mean, 'expert_std': self.save_expert_std}

        print("DAgger iterations finished!")
        print(dagger_results)

    def test(self, load_file_name):
        self.build_test_graph(self.policy_class)
        with tf.Session() as sess:
            self.load_policy(load_file_name)

            for i in range(self.num_rollouts):
                obs = self.env.reset()
                done = False
                totalr = 0.
                steps = 0

                while not done:
                    feed_dict = self.policy.eval_feed_dict(obs)
                    # run policy

                    action = self.action_hat.eval(feed_dict=feed_dict)
                    action = np.squeeze(action)
                    self.policy.print_results(obs, action)

                    action = self.env.expert_action()
                    obs, r, done, _ = self.env.step(action)
                    totalr += r
                    steps += 1

                    if steps >= self.max_steps:
                        break

    def build_graph(self, policy_class):
        with tf.variable_scope("policy"):
            self.policy = policy_class(self.dir_name)
            self.action_hat = self.policy.get_output()
            self.loss = self.policy.get_loss()
            self.train_step_op = tf.train.AdamOptimizer(learning_rate=1.0e-5).minimize(self.loss)

    def build_test_graph(self, policy_class):
        with tf.variable_scope("policy"):
            self.policy = policy_class(None)
            self.action_hat  = self.policy.get_output()

    def save_policy(self, fname):
        if fname == None:
            print("no policy save file given")
            return

        print("saving {}".format(fname))
        fname += "/"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(tf.get_default_session(), fname)

    def load_policy(self, fname):
        print("loading {}".format(fname))
        fname += "/"
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(tf.get_default_session(), fname)

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
                if path is not None:
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

    def train_dataset(self, dataset = None):
        if dataset is None:
            dataset = self.policy.get_dataset(self.samples)
            dataset = dataset.shuffle(len(self.samples))

        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(self.batch_size)
        return dataset


    def train_step(self, dataset = None):
        dataset = self.train_dataset(dataset)
        iterator = dataset.make_initializable_iterator()
        get_next = iterator.get_next()

        step = 0

        sess = tf.get_default_session()
        # do we need to reset adam at this point?

        sess.run(iterator.initializer)
        while True:
            try:
                batch = sess.run(get_next)
                self.policy.print_batch(batch)
                feed_dict = self.policy.loss_feed_dict(batch)
                _, loss = sess.run([self.train_step_op, self.loss], feed_dict=feed_dict)
                step += 1
                if (step % self.report_frequency == 0):
                    print ("train step {} objective batch loss {}".format(step, loss))

                if (step % self.save_frequency == 0):
                    self.save_policy(self.save_file_name)

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
                feed_dict = self.policy.eval_feed_dict(obs)
                # run policy

                action = self.action_hat.eval(feed_dict=feed_dict)
                action = np.squeeze(action)
                expert_action = self.env.expert_action()

                # data aggregation
                path = self.add_sample(obs, expert_action, phase=iter, rollout=i, step=steps)
                if path is not None:
                    self.env.save_cams(path)

                # take action from policy
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
