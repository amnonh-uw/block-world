from dist_world import DistworldEnv as make

def create_env(env_id, client_id, remotes, **kwargs):
    return make()