"""
CartPole Policy Network - Train and Export to ONNX

This script demonstrates:
1. Training a simple policy network for CartPole-v1
2. Exporting the trained policy to ONNX
3. Validating the exported model

The ONNX model can then be loaded by the C++ InferenceEngine
for real-time policy inference.

Usage:
    python train_cartpole.py --episodes 500 --export

Architecture:
    State (4 dims) → Dense(64) → ReLU → Dense(64) → ReLU → Actions (2 dims)
    
Output:
    - models/cartpole_policy.onnx (policy network)
    - Softmax applied to get action probabilities
"""

import argparse
import numpy as np
import os

# Check for required packages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("⚠️ Gymnasium/Gym not available. Install with: pip install gymnasium")


class PolicyNetwork(nn.Module):
    """
    Simple feed-forward policy network for CartPole
    
    Input: State vector (4 dimensions)
        - Cart position
        - Cart velocity  
        - Pole angle
        - Pole angular velocity
    
    Output: Action logits (2 dimensions)
        - Action 0: Push cart left
        - Action 1: Push cart right
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass - returns action logits (NOT probabilities)
        Softmax is applied externally for numerical stability
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_action(self, state):
        """Sample an action from the policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class REINFORCETrainer:
    """
    REINFORCE algorithm (vanilla policy gradient)
    
    Update rule:
        θ = θ + α * ∇_θ log π(a|s) * G_t
    
    Where G_t is the discounted return from time t
    """
    
    def __init__(self, policy, lr=0.01, gamma=0.99):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
    def store_outcome(self, log_prob, reward):
        """Store step outcome for later update"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        
    def update(self):
        """
        Update policy using REINFORCE
        
        Key insight: We weight gradients by returns, so good actions
        become more likely and bad actions become less likely.
        """
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (reduces variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)  # Negative because we maximize
        
        # Update policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode storage
        self.log_probs = []
        self.rewards = []
        
        return loss.item()


def train_cartpole(episodes=500, render=False):
    """Train CartPole policy using REINFORCE"""
    
    if not TORCH_AVAILABLE or not GYM_AVAILABLE:
        print("❌ Cannot train: missing dependencies")
        return None
    
    # Create environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # Create policy and trainer
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=64)
    trainer = REINFORCETrainer(policy, lr=0.01, gamma=0.99)
    
    # Training loop
    episode_rewards = []
    solved = False
    
    print("🎮 Training CartPole policy...")
    print("=" * 50)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob = policy.get_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store outcome
            trainer.store_outcome(log_prob, reward)
            episode_reward += reward
            state = next_state
        
        # Update policy after episode
        loss = trainer.update()
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:.1f} | Loss: {loss:.4f}")
            
            # Check if solved (avg reward > 195 over last 100 episodes)
            if len(episode_rewards) >= 100:
                if np.mean(episode_rewards[-100:]) >= 195:
                    print(f"\n🎉 Solved at episode {episode + 1}!")
                    solved = True
                    break
    
    env.close()
    
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print("=" * 50)
    print(f"Training complete. Final avg reward: {final_avg:.1f}")
    
    return policy


def export_to_onnx(policy, output_path="models/cartpole_policy.onnx"):
    """Export trained policy to ONNX format"""
    
    if not TORCH_AVAILABLE:
        print("❌ Cannot export: PyTorch not available")
        return False
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set to evaluation mode
    policy.eval()
    
    # Create dummy input (state vector)
    dummy_input = torch.randn(1, 4)  # Batch size 1, state dim 4
    
    # Export to ONNX
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_logits'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Exported policy to: {output_path}")
    
    # Validate export
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("✅ ONNX model validation passed")
        
        # Print model info
        print(f"   Input: {model.graph.input[0].name} {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"   Output: {model.graph.output[0].name} {[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]}")
        
    except ImportError:
        print("⚠️ ONNX not installed, skipping validation")
    except Exception as e:
        print(f"⚠️ ONNX validation warning: {e}")
    
    return True


def test_policy(policy, episodes=10, render=True):
    """Test trained policy"""
    
    if not TORCH_AVAILABLE or not GYM_AVAILABLE:
        print("❌ Cannot test: missing dependencies")
        return
    
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    policy.eval()
    
    print(f"\n🎮 Testing policy for {episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = policy(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()
    print(f"\nAverage reward: {np.mean(total_rewards):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Train CartPole policy and export to ONNX')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--export', action='store_true', help='Export to ONNX after training')
    parser.add_argument('--test', action='store_true', help='Test policy after training')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--output', type=str, default='models/cartpole_policy.onnx', help='ONNX output path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  CartPole Policy Training - REINFORCE Algorithm")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Export: {args.export}")
    print(f"  Output: {args.output}")
    print("=" * 60)
    
    # Train policy
    policy = train_cartpole(episodes=args.episodes, render=args.render)
    
    if policy is None:
        print("❌ Training failed")
        return
    
    # Export to ONNX
    if args.export:
        export_to_onnx(policy, args.output)
    
    # Test policy
    if args.test:
        test_policy(policy, episodes=5, render=args.render)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
