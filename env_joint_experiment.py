import gym
from gym import register

import safety_gym

import lava_passage
from lava_experiment import Wrapper


def create_env(args):
    safety_benchmark_robot_list = ['point', 'car', 'doggo']
    safety_benchmark_task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    if args.env_name in safety_benchmark_task_list and args.robot in safety_benchmark_robot_list:
        task = args.env_name.capitalize()
        robot = args.robot.capitalize()
        environment = gym.make('Safexp-' + robot + task + '-v0')
    elif args.env_name == 'lava':
        # version = 0
        # environment = args.env_name + '-v{}'.format(version)
        # entry_point = "lava_passage" + ":{}".format('LavaPassageEnv')
        # # entry_point="envs:MyDoorKeyEnv"
        # register(id=environment, entry_point=entry_point)

        environment = Wrapper(lava_passage.LavaPassageEnv())
        environment.seed(args.seed)
        #     environment = gym.make(env_id)
        # environment.seed(args.seed)
    # elif args.task in lyp_benchmark_task_list:
    #     environment = utils.create_env(args)
    else:
        raise ValueError('Invalid task or robot.')
    return environment