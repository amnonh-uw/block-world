from dagger import Dagger

def learn(env, policy):
    dagger = Dagger(env)
    dagger.learn(policy)
