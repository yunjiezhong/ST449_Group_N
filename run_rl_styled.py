''' Based on the example/run_rl.py file in the RLcard project, 
this is used to train agents with specific styles using NFSP based on playing against a chosen opponent.
'''
import os
import argparse
import json

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.styled_nfsp_agent import StyledNFSPAgent, load_style_config
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


def create_opponent(opponent_type, env, device, opponent_path=None):
    if opponent_type == 'random':
        return RandomAgent(num_actions=env.num_actions), 'random'
    elif opponent_type == 'checkpoint' and opponent_path:
        model_name = os.path.splitext(os.path.basename(opponent_path))[0]
        
        # debug:weights_only=False is needed for loading custom classes (NFSPAgent, StyledNFSPAgent)
        # debug:Use string 'cpu' or 'cuda:0' to avoid device deserialization issues
        map_loc = 'cuda:0' if device == torch.device('cuda') else 'cpu'
        loaded = torch.load(opponent_path, map_location=map_loc, weights_only=False)
        
        if isinstance(loaded, dict):
            if loaded.get('agent_type') == 'StyledNFSPAgent':
                opponent = StyledNFSPAgent.from_checkpoint(loaded)
            else:
                from rlcard.agents import NFSPAgent
                opponent = NFSPAgent.from_checkpoint(loaded)
        else:
            opponent = loaded
        
        # debug: Properly move all model components to the target device
        # debug: Check if it's a DQNAgent (has q_estimator directly)
        if hasattr(opponent, 'q_estimator') and hasattr(opponent, 'target_estimator'):
            # Direct DQNAgent
            opponent.device = device
            opponent.q_estimator.device = device
            opponent.q_estimator.qnet = opponent.q_estimator.qnet.to(device)
            opponent.q_estimator.qnet.eval()
            opponent.target_estimator.device = device
            opponent.target_estimator.qnet = opponent.target_estimator.qnet.to(device)
            opponent.target_estimator.qnet.eval()
        else:
            # NFSP or StyledNFSP agent
            if hasattr(opponent, 'set_device'):
                opponent.set_device(device)
            if hasattr(opponent, 'policy_network'):
                opponent.policy_network = opponent.policy_network.to(device)
                opponent.policy_network.eval()
            if hasattr(opponent, '_rl_agent'):
                if hasattr(opponent._rl_agent, 'q_estimator'):
                    opponent._rl_agent.q_estimator.qnet = opponent._rl_agent.q_estimator.qnet.to(device)
                    opponent._rl_agent.q_estimator.qnet.eval()
                if hasattr(opponent._rl_agent, 'target_estimator'):
                    opponent._rl_agent.target_estimator.qnet = opponent._rl_agent.target_estimator.qnet.to(device)
                    opponent._rl_agent.target_estimator.qnet.eval()
        
        return opponent, model_name
    else:
        return RandomAgent(num_actions=env.num_actions), 'random'


def train(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)
    
    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Load style configuration
    from rlcard.agents.styled_nfsp_agent import StyledNFSPAgent, load_style_config
    style_config = load_style_config(args.style_config)
    style_name = args.style
    style_cfg = style_config['styles'][style_name]
    
    agent = StyledNFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=device,
        save_path=args.log_dir,
        save_every=args.save_every,
        style_bias=style_cfg['bias'],
        style_beta=style_cfg['beta'],
        style_name=style_name,
    )
    
    opponent, opponent_name = create_opponent(args.opponent, env, device, args.opponent_path)
    
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(opponent)
    env.set_agents(agents)
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Start training
    with Logger(log_dir) as logger:
        for episode in range(args.num_episodes):
            
            agents[0].sample_episode_policy()
            
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)
            
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agents[0].feed(ts)
            
            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                reward = tournament(env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    
    # Plot learning curve
    plot_curve(csv_path, fig_path, f'NFSP_{style_name}')
    
    # Save model with adaptive naming
    model_name = f'NFSP_{style_name}_{opponent_name}_{args.num_episodes}_{args.seed}.pth'
    save_path = os.path.join(log_dir, model_name)
    torch.save(agents[0].checkpoint_attributes(), save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Styled NFSP training for RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--style',
        type=str,
        default='agent0'
    )
    parser.add_argument(
        '--opponent',
        type=str,
        default='random',
        choices=['random', 'checkpoint']
    )
    parser.add_argument(
        '--opponent_path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--style_config',
        type=str,
        default='style_config.json'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default=''
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=500
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/styled_nfsp/'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=1000
    )
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
