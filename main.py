import copy
import glob
import os
import time
import types
from collections import deque
import csv 
import shutil

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot, get_reward_log

import sys 

import matplotlib.pyplot as plt 
from tensorboardX import SummaryWriter
from tqdm import tqdm 


args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes


try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def create_checkpoint(agent, envs, args): 

    path = './trained_models/{}'.format(args.run_id)
    try: 
        os.makedirs(path)
    except OSError:
        pass 

    torch.save(agent.state_dict(), path + '/model_state_dict')
    with open(path + '/env.csv', 'w') as file: 
        writer = csv.writer(file)

        writer.writerow(envs.venv.ob_rms.mean)
        writer.writerow(envs.venv.ob_rms.var)

def global_rew_to_viz(writer, last_index): 

    try: 
        tx, ty = get_reward_log(args.log_dir)
        if tx != None and ty != None: 
            max_index = len(tx)
            for ind_iter in range(last_index, max_index): 
                writer.add_scalar('Reward',ty[ind_iter], tx[ind_iter])
            last_index = max_index
    except IOError:
        pass

    return last_index


def end_episode_to_viz(writer, info, info_num, epi): 

    length = info['episode']['l']
    
    writer.add_scalars('EpisodeLength', {'PS{}'.format(info_num) :length}, epi)

def losses_to_viz(writer, losses, it): 
    for k in losses.keys(): 
        writer.add_scalar("Metrics/{}".format(k), losses[k], it)

def main():

    print('Preparing parameters')

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print('Creating envs: {}'.format(args.env_name))


    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    # input(envs)
    print('Creating network')
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)


    print('Initializing PPO')
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                        args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)

    print('Memory')
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)


    num_episodes = [0 for _ in range(args.num_processes)]


    if args.run_id == "debug": 
        try: 
            shutil.rmtree('./runs/debug')
        except: 
            pass 

    writer = SummaryWriter("./runs/{}".format(args.run_id))
    with open('./runs/{}/recap.txt'.format(args.run_id), 'w') as file: 
        file.write(str(actor_critic))


    last_index = 0
    
    print('Starting ! ')

    start = time.time()
    for j in tqdm(range(num_updates)):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step], rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info_num, info in enumerate(infos):
                if(info_num == 0): 
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        end_episode_to_viz(writer, info, info_num, num_episodes[info_num])
                        num_episodes[info_num] += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        losses = agent.update(rollouts)
        rollouts.after_update()

        losses_to_viz(writer, losses, j)
        create_checkpoint(actor_critic, envs, args)
        last_index = global_rew_to_viz(writer, last_index)

if __name__ == "__main__":
    main()
