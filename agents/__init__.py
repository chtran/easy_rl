class Agent:
    def __init__(self, env, debug=False):
        self.env = env
        self.debug = debug

    def get_action(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


from .ddpg import *
from .policy_gradient import *


def get_agent_class(agent_id):
    if agent_id == 'DDPG':
        return DDPGAgent
    elif agent_id == 'PolicyGradient':
        return PolicyGradientAgent
    raise ValueError("Unsupported agent")
