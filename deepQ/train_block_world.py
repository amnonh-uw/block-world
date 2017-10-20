import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines import deepq
from block_world import BlockWorldEnv as make

def main():

    """
        q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.

    """

    def q_func(observation_in, num_actions, scope, reuse=False):
        num_actions = int(num_actions)
        with tf.variable_scope(scope, reuse=reuse):
            out = observation_in
            out = layers.fully_connected(out, 256)
            out = layers.fully_connected(out, 256)
            out = layers.fully_connected(out, num_actions)

        return out

    env = make(pos_unit=0.1, rot_unit=1)

    act = deepq.learn(
        env,
        q_func=q_func,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    act.save("blockworld_model.pkl")
    env.close()

if __name__ == '__main__':
    main()
