class Agent:
    def __init__(self, env, debug=False):
        self.env = env
        self.debug = debug

    def get_action(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


from .ddpg import *


def get_agent_class(agent_id):
    if agent_id == 'DDPG':
        return DDPGAgent
    raise ValueError("Unsupported agent")
