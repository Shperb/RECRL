#!/usr/bin/env python
import gym
import safety_gym
import safe_rl
import os
import numpy as np
import random
import torch
import shutil

from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

from cmdps_via_bvf.agents.lyp_ppo import LyapunovPPOAgent
from cmdps_via_bvf.agents.lyp_sarsa_agent import LypSarsaAgent
from cmdps_via_bvf.agents.safe_ppo import SafePPOAgent
from cmdps_via_bvf.agents.safe_sarsa_agent import SafeSarsaAgent
from cmdps_via_bvf.agents.sarsa_agent import SarsaAgent
from cmdps_via_bvf.agents.target_agents.target_bvf_ppo import TargetBVFPPOAgent
from cmdps_via_bvf.agents.target_agents.target_lyp_ppo import TargetLypPPOAgent
from cmdps_via_bvf.common.utils import get_filename

from cost_limit_curriculum import StaticCostLimitCurriculum, DynamicCurriculum
from env_joint_experiment import create_env
from static_cost_decay_functions import LinearDecay, ExponentialDecay, CosineDecay, NoDecay


def main(args):

    # Verify experiment

    safety_benchmark_algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    lyp_benchmark_algo_list = ["bvf-ppo", "lyp-ppo", "sarsa", "bvf-sarsa", "lyp-sarsa"]

    if args.decay_type == 'dynamic':
        cost_curriclum = DynamicCurriculum(args.d_d, d_t=args.d_t, decay_end=args.decay_end, window=10, cost_delta_threshold=0.1, return_delta_threshold=0.1,initial_noise = 0.3)

    elif args.decay_type == 'linear':
        cost_curriclum = StaticCostLimitCurriculum(args.d_d,LinearDecay(),d_t=args.d_t,decay_end=args.decay_end)

    elif args.decay_type == 'exponential':
        cost_curriclum = StaticCostLimitCurriculum(args.d_d,ExponentialDecay(),d_t=args.d_t,decay_end=args.decay_end)

    elif args.decay_type == 'cosine':
        cost_curriclum = StaticCostLimitCurriculum(args.d_d,CosineDecay(),d_t=args.d_t,decay_end=args.decay_end)
    else:
        cost_curriclum = StaticCostLimitCurriculum(args.d_d,NoDecay(),d_t=args.d_t,decay_end=args.decay_end)


    environment = create_env(args)

    algo = args.agent.lower()
    if algo in safety_benchmark_algo_list:
        task = args.env_name.capitalize()
        robot = args.robot.capitalize()

        # Hyperparameters
        exp_name = algo + '_' + robot + task
        num_steps = args.num_steps
        epochs = int(num_steps / args.steps_per_epoch)
        save_freq = 50
        target_kl = 0.01


        # Fork for parallelizing
        if (args.cpu > 1):
            mpi_fork(args.cpu)

        # Prepare Logger
        exp_name = exp_name or (algo + '_' + robot.lower() + task.lower()) +'_' + args.decay_type
        logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

        # Algo and Env
        algo = eval('safe_rl.'+algo)

        algo(env_fn=lambda: environment,
             ac_kwargs=dict(
                 hidden_sizes=(256, 256),
                ),
             epochs=epochs,
             steps_per_epoch=args.steps_per_epoch,
             save_freq=save_freq,
             target_kl=target_kl,
             cost_lim=args.d_d,
             seed=args.seed,
             logger_kwargs=logger_kwargs,
             cost_lim_curriculum=cost_curriclum
             )
    elif algo in lyp_benchmark_algo_list:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        # pytorch multiprocessing flag
        torch.set_num_threads(1)

        # check the device here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get the filename
        name = get_filename(args)
        args.out = os.path.join(args.out, args.env_name, args.agent, name)
        tb_log_dir = os.path.join(args.log_dir, args.env_name, args.agent, name, 'tb_logs')
        if args.reset_dir:
            shutil.rmtree(args.out, ignore_errors=True)  # delete the results dir
            shutil.rmtree(tb_log_dir, ignore_errors=True)  # delete the tb dir

        os.makedirs(args.out, exist_ok=True)
        os.makedirs(tb_log_dir, exist_ok=True)

        # don't use tb on cluster
        tb_writer = None

        # print the dir in the beginning
        print("Log dir", tb_log_dir)
        print("Out dir", args.out)

        if args.agent == "bvf-ppo":
            if args.target:
                agent = TargetBVFPPOAgent(args, environment,cost_lim_curriculum=cost_curriclum)
            else:
                agent = SafePPOAgent(args, environment, writer=tb_writer,cost_lim_curriculum=cost_curriclum)
        elif args.agent == "lyp-ppo":
            if args.target:
                agent = TargetLypPPOAgent(args, environment,cost_lim_curriculum=cost_curriclum)
            else:
                agent = LyapunovPPOAgent(args, environment,cost_lim_curriculum=cost_curriclum)
        elif args.agent == "sarsa":
            agent = SarsaAgent(args, environment, writer=tb_writer)
        elif args.agent == "bvf-sarsa":
            agent = SafeSarsaAgent(args, environment, writer=tb_writer,cost_lim_curriculum=cost_curriclum)
        elif args.agent == "lyp-sarsa":
            agent = LypSarsaAgent(args, environment, writer=tb_writer,cost_lim_curriculum=cost_curriclum)
        else:
            raise Exception("Not implemented yet")

        # start the run process here
        agent.run()

    else:
        raise ValueError('Invalid algorithm.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--env_name', type=str, default='Goal1')
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--decay_type', type=str, default='none')
    parser.add_argument('--d_t', type=float, default=1)
    parser.add_argument('--decay_end', type=float, default=1)

    #args from lyp repo
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--d_d', type=float, default=25.0, help="the threshold for safety")

    # Actor Critic arguments goes here
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help="learning rate")
    parser.add_argument('--target-update-steps', type=int, default=int(1e4),
                        help="number of steps after to train the agent")
    parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--updates-per-step', type=int, default=1, help='model updates per simulator step (default: 1)')
    parser.add_argument('--tau', type=float, default=0.001, help='soft update rule for target netwrok(default: 0.001)')

    # PPO arguments go here
    parser.add_argument('--num-envs', type=int, default=10, help='the num of envs to gather data in parallel')
    parser.add_argument('--ppo-updates', type=int, default=1, help='num of ppo updates to do')
    parser.add_argument('--gae', type=float, default=0.95, help='GAE coefficient')
    parser.add_argument('--clip', type=float, default=0.2, help='clipping param for PPO')
    parser.add_argument('--traj-len', type=int, default=10, help="the maximum length of the trajectory for an update")
    parser.add_argument('--early-stop', action='store_true',
                        help="early stop pi training based on target KL ")

    # Optmization arguments
    parser.add_argument('--lr', type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument('--adam-eps', type=float, default=0.95, help="momenturm for RMSProp")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of minibatch for ppo/ ddpg update')

    # Safety params
    parser.add_argument('--cost-reverse-lr', type=float, default=5e-4,
                        help="reverse learning rate for reviewer")
    parser.add_argument('--cost-q-lr', type=float, default=5e-4,
                        help="reverse learning rate for critic")
    parser.add_argument('--cost-sg-coeff', type=float, default=0.0,
                        help="the coeeficient for the safe guard policy, minimizes the cost")
    parser.add_argument('--prob-alpha', type=float, default=0.6,
                        help="the kappa parameter for the target networks")
    parser.add_argument('--target', action='store_true',
                        help="use the target network based implementation")

    # Training arguments
    parser.add_argument('--num-steps', type=int, default=int(1e6),
                        help="number of steps to train the agent")
    parser.add_argument('--steps-per-epoch', type=int, default=int(1e4),
                        help="number of steps to train the agent")
    parser.add_argument('--num-episodes', type=int, default=int(1e4),
                        help="number of episodes to train the agent")
    parser.add_argument('--max-ep-len', type=int, default=int(15),
                        help="number of steps in an episode")

    # Evaluation arguments
    parser.add_argument('--eval-every', type=float, default=1000,
                        help="eval after these many steps")
    parser.add_argument('--eval-n', type=int, default=1,
                        help="average eval results over these many episodes")

    # Experiment specific
    parser.add_argument('--gpu', action='store_true', help="use the gpu and CUDA")
    parser.add_argument('--log-mode-steps', action='store_true',
                        help="changes the mode of logging w.r.r num of steps instead of episodes")
    parser.add_argument('--log-every', type=int, default=100,
                        help="logging schedule for training")
    parser.add_argument('--checkpoint-interval', type=int, default=1e5,
                        help="when to save the models")
    parser.add_argument('--out', type=str, default='/tmp/safe/models/')
    parser.add_argument('--log-dir', type=str, default="/tmp/safe/logs/")
    parser.add_argument('--reset-dir', action='store_true',
                        help="give this argument to delete the existing logs for the current set of parameters")

    args = parser.parse_args()
    main(args)