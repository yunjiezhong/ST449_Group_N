import sys
import os
import torch
import rlcard
from rlcard.agents import DQNAgent, NFSPAgent

try:
    from rlcard.agents import styled_nfsp_agent
    sys.modules['rlcard.agents.styled_nfsp_agent'] = styled_nfsp_agent
except ImportError:
    pass

class LoadedAgent:
    def __init__(self, model_path, algorithm=None, env=None, device='cpu'):
        self.use_raw = False
        
        self.model_path = model_path
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.env = env
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model path not exist: {model_path}")

        try:
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            self.checkpoint = torch.load(model_path, map_location=self.device)

        if not isinstance(self.checkpoint, dict):
            self.agent = self.checkpoint
            self._set_agent_device(self.agent)
            self._set_eval_mode(self.agent)
            return

        if algorithm is None:
            algorithm = self._detect_algorithm()
        self.algorithm = algorithm.lower()

        if self.algorithm == 'nfsp':
            self.agent = self._load_nfsp()
        elif self.algorithm == 'dqn':
            self.agent = self._load_dqn()
        else:
            raise ValueError(f"cant identify algorithm: {self.algorithm}")

    def _set_agent_device(self, agent):
        if hasattr(agent, 'set_device'):
            agent.set_device(self.device)
        elif hasattr(agent, 'device'):
            agent.device = self.device
        if hasattr(agent, 'q_estimator') and hasattr(agent.q_estimator, 'qnet'):
            agent.q_estimator.qnet.to(self.device)
    
    def _set_eval_mode(self, agent):
        if hasattr(agent, 'policy_network'):
            agent.policy_network.eval()
        
        if hasattr(agent, 'q_estimator') and hasattr(agent.q_estimator, 'qnet'):
            agent.q_estimator.qnet.eval()
        
        if hasattr(agent, 'target_estimator') and hasattr(agent.target_estimator, 'qnet'):
            agent.target_estimator.qnet.eval()
        
        if hasattr(agent, 'q_estimator') and hasattr(agent.q_estimator, 'qnet'):
            agent.q_estimator.qnet.eval()
        
        if hasattr(agent, '_rl_agent'):
            if hasattr(agent._rl_agent, 'q_estimator') and hasattr(agent._rl_agent.q_estimator, 'qnet'):
                agent._rl_agent.q_estimator.qnet.eval()
            if hasattr(agent._rl_agent, 'target_estimator') and hasattr(agent._rl_agent.target_estimator, 'qnet'):
                agent._rl_agent.target_estimator.qnet.eval()
        
        if hasattr(agent, 'policy_network'):
            agent.policy_network.eval()

    def _get_env_info(self):
        if self.env is None:
            raise ValueError(f"loading {self.model_path}  need env")

        if hasattr(self.env, 'num_actions'):
            action_num = self.env.num_actions
        elif hasattr(self.env, 'action_num'):
            action_num = self.env.action_num
        else:
            try:
                action_num = self.env.game.get_num_actions()
            except:
                raise AttributeError("cant find env ")

        state_shape = self.env.state_shape
        if isinstance(state_shape, list) and isinstance(state_shape[0], list):
            state_shape = state_shape[0]
            
        return action_num, state_shape

    def _detect_algorithm(self):
        if 'reservoir_buffer' in self.checkpoint or 'policy_network' in self.checkpoint:
            return 'nfsp'
        return 'dqn'

    def _load_nfsp(self):
        self.checkpoint['device'] = self.device
        
        if 'rl_agent' in self.checkpoint:
             try:
                 from rlcard.agents.styled_nfsp_agent import StyledNFSPAgent
                 agent = StyledNFSPAgent.from_checkpoint(self.checkpoint)
                 self._set_agent_device(agent)
                 self._set_eval_mode(agent)
                 return agent
             except (ImportError, AttributeError):
                 pass
             
        print(f"NFSP: {self.model_path}")
        action_num, state_shape = self._get_env_info()
            
        agent = NFSPAgent(
            num_actions=action_num,
            state_shape=state_shape,
            hidden_layers_sizes=[64, 64],
            q_mlp_layers=[64, 64],
            device=self.device
        )
        
        policy_data = None
        if 'policy_network' in self.checkpoint:
            policy_data = self.checkpoint['policy_network']
        elif 'model_state_dict' in self.checkpoint:
            policy_data = self.checkpoint['model_state_dict']
        else:
            policy_data = self.checkpoint

        if policy_data is not None:
            try:
                if isinstance(policy_data, dict) and 'mlp' in policy_data:
                    agent.policy_network.load_state_dict(policy_data['mlp'])
                else:
                    agent.policy_network.load_state_dict(policy_data)
            except RuntimeError as e:
                print(f"{e}")
                try:
                    if isinstance(policy_data, dict) and 'mlp' in policy_data:
                        agent.policy_network.load_state_dict(policy_data['mlp'], strict=False)
                    else:
                        agent.policy_network.load_state_dict(policy_data, strict=False)
                except Exception:
                    print("loading fail")

        agent.sample_episode_policy = lambda: None
        agent._mode = 'average_policy'
        
        self._set_eval_mode(agent)
        
        return agent

    def _load_dqn(self):
        action_num, state_shape = self._get_env_info()
        agent = DQNAgent(
            num_actions=action_num,
            state_shape=state_shape,
            mlp_layers=[64, 64], 
            device=self.device
        )
        agent.load(self.checkpoint)
        
        self._set_eval_mode(agent)
        
        return agent

    def step(self, state):
        if hasattr(self.agent, 'eval_step'):
             action, _ = self.agent.eval_step(state)
        else:
             action = self.agent.step(state)
        return action

    def eval_step(self, state):
        return self.agent.eval_step(state)
