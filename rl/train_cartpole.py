#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                    CARTPOLE POLICY TRAINING - REINFORCE ALGORITHM
================================================================================
                    Educational Implementation with Detailed Comments
================================================================================

FILE: train_cartpole.py
PURPOSE: Train a neural network policy for the CartPole environment using the
         REINFORCE algorithm, then export to ONNX for C++ inference.

WHAT IS CARTPOLE?
-----------------
CartPole is a classic reinforcement learning benchmark problem:
- A pole is attached to a cart that moves along a frictionless track
- Goal: Keep the pole balanced upright as long as possible
- Control: Apply force to move the cart left or right
- Episode ends when:
  * Pole angle exceeds ±12 degrees
  * Cart position exceeds ±2.4 (falls off track)
  * 500 timesteps reached (success!)

STATE SPACE (4 continuous values):
    [0] Cart Position:       -4.8 to 4.8 (track boundaries)
    [1] Cart Velocity:       -∞ to ∞ (how fast cart is moving)
    [2] Pole Angle:          -24° to 24° (radians, ~±0.42)
    [3] Pole Angular Velocity: -∞ to ∞ (how fast pole is falling)

ACTION SPACE (2 discrete actions):
    0: Push cart LEFT  (apply force in negative direction)
    1: Push cart RIGHT (apply force in positive direction)

REWARD:
    +1 for every timestep the pole remains balanced
    Maximum episode reward: 500 (if you balance for all 500 steps)
    "Solved" when average reward ≥ 195 over 100 consecutive episodes

WHAT IS REINFORCE?
------------------
REINFORCE is the simplest policy gradient algorithm:
- Directly parameterizes a policy π(a|s; θ) (probability of action given state)
- Uses Monte Carlo sampling to estimate gradient
- Updates policy in the direction that increases expected reward

The core insight:
    If an action led to HIGH reward → make it MORE likely
    If an action led to LOW reward → make it LESS likely

The math (Policy Gradient Theorem):
    ∇_θ J(θ) = E_π [ ∇_θ log π(a|s; θ) * G_t ]
    
Where:
    - J(θ) is the expected cumulative reward
    - π(a|s; θ) is the policy (neural network)
    - G_t is the return (discounted sum of future rewards from time t)
    - ∇_θ means "gradient with respect to parameters θ"

In plain English: "Adjust weights to make actions that led to good outcomes
                  more probable in similar states."

WHY ONNX EXPORT?
----------------
Once trained in Python (easy experimentation), we export to ONNX so that:
1. C++ InferenceEngine can load it for fast real-time inference
2. No Python dependency in production deployment
3. Can run on edge devices, embedded systems, game engines
4. Standardized format works across frameworks

NEURAL NETWORK ARCHITECTURE:
-----------------------------
    Input Layer:  4 neurons (state dimensions)
         ↓
    Hidden Layer 1: 64 neurons + ReLU activation
         ↓
    Hidden Layer 2: 64 neurons + ReLU activation
         ↓
    Output Layer: 2 neurons (action logits)
         ↓
    Softmax (applied externally for numerical stability)
         ↓
    Action Probabilities: [P(left), P(right)]

This is a small network (~5K parameters) suitable for the simple CartPole task.
More complex environments would need deeper/wider networks.

USAGE EXAMPLES:
---------------
    # Basic training (500 episodes)
    python train_cartpole.py
    
    # Train and export to ONNX
    python train_cartpole.py --episodes 500 --export
    
    # Train, export, and test
    python train_cartpole.py --episodes 1000 --export --test
    
    # With visualization
    python train_cartpole.py --render --test

OUTPUTS:
--------
    - models/cartpole_policy.onnx : The trained policy in ONNX format
    - Training progress printed to console
    - Test results showing average reward

Author: Multi-Language AI System Educational Project
================================================================================
"""

# ==============================================================================
#                              IMPORTS AND SETUP
# ==============================================================================

# --- Standard Library Imports ------------------------------------------------
import argparse  # For parsing command-line arguments (--episodes, --export, etc.)
                 # argparse is Python's built-in argument parser
                 # It automatically generates help messages and handles type conversion

import numpy as np  # NumPy: Numerical Python
                    # Used for: array operations, statistical calculations (mean, std)
                    # NumPy is the foundation of scientific Python
                    # Key functions we use: np.mean(), np.array(), reversed operations

import os  # Operating system interface
           # Used for: file paths, directory creation
           # os.makedirs() - create directories recursively
           # os.path.dirname() - extract directory from path

# ==============================================================================
#                           PYTORCH IMPORT (OPTIONAL)
# ==============================================================================
# PyTorch is a deep learning framework. We wrap imports in try/except to
# provide a helpful error message if it's not installed.

try:
    # --- PyTorch Core --------------------------------------------------------
    import torch  # Main PyTorch module
                  # torch.Tensor - multi-dimensional arrays with GPU support
                  # torch.FloatTensor - 32-bit floating point tensor
                  # torch.no_grad() - disable gradient computation for inference
                  # torch.randn() - random normal distribution tensor
    
    # --- Neural Network Module -----------------------------------------------
    import torch.nn as nn  # Neural network building blocks
                           # nn.Module - base class for all neural network modules
                           # nn.Linear - fully connected (dense) layer
                           #   Linear(in, out) : y = x @ W.T + b
                           # All learnable parameters automatically tracked
    
    # --- Optimizers ----------------------------------------------------------
    import torch.optim as optim  # Optimization algorithms
                                 # optim.Adam - Adaptive Moment Estimation
                                 #   Combines momentum + adaptive learning rates
                                 #   Generally best default optimizer
                                 # optim.SGD - Stochastic Gradient Descent
                                 #   Simpler but requires more tuning
    
    # --- Functional Interface ------------------------------------------------
    import torch.nn.functional as F  # Functional (stateless) operations
                                     # F.relu(x) - Rectified Linear Unit: max(0, x)
                                     #   Introduces non-linearity
                                     #   Most popular activation function
                                     # F.softmax(x, dim) - Convert logits to probabilities
                                     #   softmax(x_i) = exp(x_i) / Σ exp(x_j)
                                     #   Output sums to 1.0
    
    # --- Probability Distributions -------------------------------------------
    from torch.distributions import Categorical  # Categorical distribution
                                                  # For sampling discrete actions
                                                  # Takes probabilities, returns samples
                                                  # .sample() - draw random action
                                                  # .log_prob(x) - log probability of x
                                                  # Log probs are numerically stable
    
    # Flag indicating PyTorch is available
    TORCH_AVAILABLE = True
    
except ImportError:
    # ImportError raised when module not found
    # Set flag to False so we can handle this gracefully
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")
    # User should run: pip install torch
    # Or for GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu118

# ==============================================================================
#                      GYMNASIUM/GYM IMPORT (OPTIONAL)
# ==============================================================================
# Gymnasium (formerly OpenAI Gym) provides the CartPole environment.
# We try both names for compatibility with older installations.

try:
    # Gymnasium is the maintained fork (recommended)
    import gymnasium as gym  # Reinforcement learning environments
                             # gym.make('CartPole-v1') - create environment
                             # env.reset() - reset to initial state
                             # env.step(action) - take action, get next state
                             # env.close() - clean up resources
    GYM_AVAILABLE = True
    
except ImportError:
    # Fall back to older "gym" package name
    try:
        import gym  # Original OpenAI Gym (deprecated but still works)
        GYM_AVAILABLE = True
        
    except ImportError:
        # Neither package is installed
        GYM_AVAILABLE = False
        print("⚠️ Gymnasium/Gym not available. Install with: pip install gymnasium")
        # User should run: pip install gymnasium


# ==============================================================================
#                           POLICY NETWORK CLASS
# ==============================================================================
# The policy network is a neural network that takes in the state (what the
# agent observes) and outputs action probabilities (what to do).
#
# This is the "brain" of our RL agent!

class PolicyNetwork(nn.Module):
    """
    Simple Feed-Forward Policy Network for CartPole
    ================================================
    
    WHAT IS A POLICY?
    -----------------
    In reinforcement learning, a "policy" π(a|s) is a mapping from states to
    actions. It tells the agent what to do in each situation:
    
        π(a|s) = P(action = a | state = s)
    
    For CartPole:
        π(left|state) = probability of pushing left
        π(right|state) = probability of pushing right
        
    These probabilities sum to 1.0 (we must do something!)
    
    WHY A NEURAL NETWORK?
    ---------------------
    Instead of a lookup table (impossible for continuous states), we use a
    neural network to approximate the policy:
    
        π(a|s; θ) ≈ NeuralNetwork(s; θ)
    
    Where θ (theta) represents all the network weights and biases.
    
    The network learns to map states → action probabilities through training.
    
    NETWORK ARCHITECTURE:
    ---------------------
    Input (state):        4 neurons
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Linear Layer 1:    4 → 64 neurons      │  fc1: y = Wx + b
    │  ReLU Activation:   max(0, x)           │  Non-linearity
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Linear Layer 2:    64 → 64 neurons     │  fc2: y = Wx + b
    │  ReLU Activation:   max(0, x)           │  Non-linearity
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Linear Layer 3:    64 → 2 neurons      │  fc3: y = Wx + b (logits)
    └─────────────────────────────────────────┘
           │
           ▼
    Output (logits):      2 neurons
    
    WHAT ARE LOGITS?
    ----------------
    "Logits" are the raw (unnormalized) output of the network before softmax.
    They can be any real number (-∞ to +∞).
    
    We apply softmax EXTERNALLY (not in the network) because:
    1. Numerical stability: log(softmax(x)) can underflow
    2. Cross-entropy loss works directly with logits
    3. ONNX export is cleaner
    
    PARAMETER COUNT:
    ----------------
    fc1: 4 * 64 + 64 = 320 weights + biases
    fc2: 64 * 64 + 64 = 4160 weights + biases
    fc3: 64 * 2 + 2 = 130 weights + biases
    Total: ~4,610 parameters (very small network!)
    
    WHY ReLU?
    ---------
    ReLU (Rectified Linear Unit) is the activation function:
    
        ReLU(x) = max(0, x)
    
    Without activation functions, stacking linear layers is pointless:
        y = W2(W1 * x + b1) + b2 = W_combined * x + b_combined
        
    ReLU introduces non-linearity, allowing the network to learn complex
    patterns. Other options: tanh, sigmoid, LeakyReLU, GELU, etc.
    
    INPUT DETAILS:
    --------------
        state[0]: Cart position     (-4.8 to 4.8)
        state[1]: Cart velocity     (unbounded, typically -2 to 2)
        state[2]: Pole angle        (~-0.42 to 0.42 radians)
        state[3]: Pole angular vel  (unbounded, typically -2 to 2)
    
    OUTPUT DETAILS:
    ---------------
        logits[0]: Score for action "push left"
        logits[1]: Score for action "push right"
        
        After softmax:
        probs[0]: P(push left)  = exp(logits[0]) / (exp(logits[0]) + exp(logits[1]))
        probs[1]: P(push right) = exp(logits[1]) / (exp(logits[0]) + exp(logits[1]))
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        """
        Initialize the PolicyNetwork.
        
        The __init__ method is called when you create a new PolicyNetwork():
            policy = PolicyNetwork()  # Calls this __init__
        
        Args:
            state_dim (int): Size of state vector (4 for CartPole)
                             - Cart position, velocity, pole angle, angular velocity
            
            action_dim (int): Number of possible actions (2 for CartPole)
                              - 0: push left, 1: push right
            
            hidden_dim (int): Number of neurons in hidden layers (64 default)
                              - Larger = more capacity but slower/overfitting risk
                              - Smaller = faster but might underfit
                              - 64 is plenty for CartPole
        """
        # Call parent class (nn.Module) constructor
        # ALWAYS do this first in PyTorch modules!
        # super() returns a temporary object of the parent class
        # This initializes internal bookkeeping (parameter tracking, hooks, etc.)
        super(PolicyNetwork, self).__init__()
        
        # --- Define the layers ---
        # nn.Linear(in_features, out_features) creates a fully connected layer
        # Internally stores:
        #   - weight: Tensor of shape (out_features, in_features)
        #   - bias: Tensor of shape (out_features,)
        # Forward pass: output = input @ weight.T + bias
        
        # First fully connected layer: state → hidden representation
        # Takes 4 inputs (state), produces 64 outputs
        # Each of 64 neurons "looks at" all 4 state values with different weights
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 4 → 64
        
        # Second fully connected layer: hidden → hidden
        # Allows network to learn more complex patterns
        # "Deep" networks (multiple layers) can learn hierarchical features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 64 → 64
        
        # Third (output) layer: hidden → action logits
        # Produces one score per possible action
        # Higher score = network thinks this action is better
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 64 → 2
        
    def forward(self, x):
        """
        Forward pass - compute action logits from state.
        
        The forward() method defines how data flows through the network.
        In PyTorch, you override this method to define your computation.
        
        When you call policy(state), PyTorch actually calls policy.forward(state)
        (plus some bookkeeping).
        
        Args:
            x (torch.Tensor): State tensor of shape (batch_size, state_dim)
                              For CartPole: (batch_size, 4)
                              
                              Batch dimension allows processing multiple states
                              at once for efficiency.
        
        Returns:
            torch.Tensor: Action logits of shape (batch_size, action_dim)
                          For CartPole: (batch_size, 2)
                          
                          These are RAW scores, NOT probabilities!
                          Apply softmax to convert to probabilities.
        
        Computation Flow:
            x:      [batch, 4]   (state)
            fc1:    [batch, 64]  (hidden, via matmul)
            relu:   [batch, 64]  (non-linearity applied)
            fc2:    [batch, 64]  (hidden, via matmul)
            relu:   [batch, 64]  (non-linearity applied)
            fc3:    [batch, 2]   (logits, raw output)
        """
        # Layer 1: Linear transformation followed by ReLU activation
        # self.fc1(x) computes: x @ fc1.weight.T + fc1.bias
        # F.relu() applies element-wise max(0, value)
        # After this: x has shape (batch, 64), values ≥ 0
        x = F.relu(self.fc1(x))
        
        # Layer 2: Another linear + ReLU
        # Each neuron in fc2 sees all 64 outputs from fc1
        # This layer can learn combinations of features from layer 1
        x = F.relu(self.fc2(x))
        
        # Output layer: Linear only (NO activation!)
        # We return raw logits because:
        # 1. Softmax will be applied separately (numerical stability)
        # 2. Some loss functions (cross-entropy) expect logits
        # 3. ONNX export: better to keep softmax separate
        logits = self.fc3(x)
        
        # Return logits (not probabilities!)
        # Shape: (batch_size, 2) for CartPole
        return logits
    
    def get_action(self, state):
        """
        Sample an action from the policy distribution.
        
        This method is used during TRAINING when we want to explore:
        - Convert state to tensor
        - Get action probabilities from network
        - SAMPLE from the distribution (stochastic!)
        - Return both the action AND its log probability
        
        Why sample instead of taking argmax?
        ------------------------------------
        During training, we NEED exploration:
        - If we always pick the "best" action, we never try alternatives
        - Random sampling ensures we explore the action space
        - Bad actions have low probability, so we mostly do good actions
        - But occasionally we try sub-optimal actions → might discover better strategy
        
        Why return log probability?
        ---------------------------
        REINFORCE update rule uses:
            gradient = log_prob(action) * return
            
        We need to know "how surprising was this action?" to compute the gradient.
        Log probabilities are more numerically stable than raw probabilities.
        
        Args:
            state (np.ndarray or list): The environment state
                                        [cart_pos, cart_vel, pole_angle, pole_vel]
        
        Returns:
            tuple: (action, log_prob)
                - action (int): 0 (left) or 1 (right)
                - log_prob (torch.Tensor): Log probability of the sampled action
                                           Used for policy gradient computation
        """
        # Convert numpy array/list to PyTorch tensor
        # torch.FloatTensor creates a 32-bit float tensor
        # .unsqueeze(0) adds batch dimension: shape [4] → [1, 4]
        # This is needed because our network expects batched input
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Forward pass: get action logits
        # Shape: [1, 2] (batch of 1, 2 actions)
        logits = self.forward(state)
        
        # Convert logits to probabilities using softmax
        # softmax(x_i) = exp(x_i) / sum(exp(x_j))
        # dim=-1 means apply softmax over last dimension (actions)
        # Shape: [1, 2], values sum to 1.0
        probs = F.softmax(logits, dim=-1)
        
        # Create a categorical (discrete) probability distribution
        # Categorical takes probabilities and allows sampling
        # This is like a weighted coin flip: higher prob actions more likely
        dist = Categorical(probs)
        
        # Sample an action from the distribution
        # With probs [0.7, 0.3]: action=0 sampled 70% of the time
        # .sample() returns a tensor, so action is a 0-d tensor
        action = dist.sample()
        
        # Return:
        # - action.item(): Convert 0-d tensor to Python int (0 or 1)
        # - dist.log_prob(action): Log probability of the sampled action
        #   If P(action) = 0.7, log_prob = log(0.7) ≈ -0.357
        #   Used in REINFORCE gradient computation
        return action.item(), dist.log_prob(action)


# ==============================================================================
#                        REINFORCE TRAINER CLASS
# ==============================================================================
# This class implements the REINFORCE algorithm, one of the simplest and most
# fundamental policy gradient methods. Understanding this is key to understanding
# more advanced algorithms like PPO, A2C, etc.

class REINFORCETrainer:
    """
    REINFORCE Algorithm Implementation (Vanilla Policy Gradient)
    =============================================================
    
    WHAT IS REINFORCE?
    ------------------
    REINFORCE (REward Increment = Nonnegative Factor × Offset Reinforcement 
    × Characteristic Eligibility) is a Monte Carlo policy gradient algorithm
    introduced by Ronald Williams in 1992.
    
    Key idea: Update policy parameters to make actions that led to HIGH returns
              more probable, and actions that led to LOW returns less probable.
    
    THE MATH (Policy Gradient Theorem):
    -----------------------------------
    We want to find policy parameters θ that maximize expected cumulative reward:
    
        J(θ) = E_π [ Σ_t r_t ]  (expected sum of rewards under policy π)
    
    The gradient of this objective is:
    
        ∇_θ J(θ) = E_π [ Σ_t ∇_θ log π(a_t | s_t; θ) * G_t ]
    
    Where:
        - π(a|s; θ) is the policy (probability of action a in state s)
        - G_t is the "return" (discounted sum of future rewards from time t)
        - ∇_θ log π means "gradient of log probability w.r.t. parameters"
    
    In plain English:
        "Move parameters in the direction that increases log probability of
         actions, weighted by how good the outcome was."
    
    WHY LOG PROBABILITY?
    --------------------
    We use log π instead of π because:
    1. ∇_θ log π = (1/π) * ∇_θ π  (chain rule)
       This means ∇_θ log π(a|s) points in direction that increases P(a|s)
    
    2. Log transforms products into sums (useful for sequential decisions)
       log(P(a1) * P(a2) * P(a3)) = log P(a1) + log P(a2) + log P(a3)
    
    3. Numerical stability: log(0.001) = -6.9 is manageable
                           whereas 0.001 can underflow
    
    WHAT IS THE RETURN G_t?
    -----------------------
    The return is the discounted sum of future rewards starting from time t:
    
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T
    
    Where γ (gamma) is the discount factor (0 < γ ≤ 1).
    
    For CartPole with γ=0.99:
        - Immediate reward: counts fully (weight = 1.0)
        - Reward 10 steps later: counts 0.99^10 ≈ 0.90 as much
        - Reward 100 steps later: counts 0.99^100 ≈ 0.37 as much
    
    Why discount?
        - Future rewards are uncertain (might not reach them)
        - Prefer rewards sooner over rewards later
        - Makes infinite-horizon problems mathematically tractable
    
    THE UPDATE RULE:
    ----------------
        θ ← θ + α * ∇_θ log π(a_t | s_t) * G_t
    
    Or in gradient descent form (PyTorch minimizes, so we negate):
        loss = -Σ_t log π(a_t | s_t) * G_t
    
    VARIANCE REDUCTION:
    -------------------
    Raw REINFORCE has high variance (noisy updates). We use two tricks:
    
    1. Return normalization:
       Instead of raw G_t, use (G_t - mean(G)) / std(G)
       This centers returns around 0, so good actions have positive weight
       and bad actions have negative weight.
    
    2. More episodes (Monte Carlo): Average over many episodes for stable updates.
    
    Advanced algorithms (A2C, PPO) add a "baseline" or "value function" to
    further reduce variance, but that's beyond this educational example.
    
    ALGORITHM PSEUDOCODE:
    ---------------------
    For each episode:
        1. Initialize state s_0
        2. For t = 0, 1, 2, ... until done:
            a. Sample action a_t ~ π(·|s_t; θ)
            b. Take action, observe reward r_t and next state s_{t+1}
            c. Store (log π(a_t|s_t), r_t)
        3. Compute returns G_t for all t
        4. Update: θ ← θ + α * Σ_t ∇_θ log π(a_t|s_t) * G_t
    """
    
    def __init__(self, policy, lr=0.01, gamma=0.99):
        """
        Initialize the REINFORCE trainer.
        
        Args:
            policy (PolicyNetwork): The neural network policy to train
                                    Must have parameters() method for optimizer
            
            lr (float): Learning rate (α in the update rule)
                        - Too high (0.1): Unstable, overshoots
                        - Too low (0.0001): Slow convergence
                        - 0.01 is a reasonable starting point for REINFORCE
                        
            gamma (float): Discount factor (γ)
                          - 0: Only care about immediate reward
                          - 1: All future rewards count equally (can diverge!)
                          - 0.99: Balance between immediate and future
                          - For CartPole, 0.99 works well
        """
        # Store reference to the policy network
        # This is the neural network we're training
        self.policy = policy
        
        # --- Create the optimizer ---
        # Optimizers implement gradient descent algorithms
        # They adjust network parameters based on computed gradients
        #
        # Adam (Adaptive Moment Estimation) is our choice because:
        # 1. Adapts learning rate per-parameter (some weights need bigger updates)
        # 2. Uses momentum (smooths out noisy gradients)
        # 3. Generally robust with default settings
        #
        # policy.parameters() returns an iterator over all trainable parameters
        # (weights and biases of fc1, fc2, fc3)
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        # Store discount factor
        # Used to compute returns: G_t = r_t + γ*G_{t+1}
        self.gamma = gamma
        
        # --- Episode storage ---
        # REINFORCE is a Monte Carlo method: we need to complete an entire
        # episode before we can compute returns and update the policy.
        #
        # We store:
        # - log_probs: log π(a_t | s_t) for each timestep
        # - rewards: r_t for each timestep
        #
        # These lists get cleared after each policy update
        self.log_probs = []  # List of torch.Tensor, one per timestep
        self.rewards = []    # List of float, one per timestep
        
    def store_outcome(self, log_prob, reward):
        """
        Store the outcome of a single timestep.
        
        Called after each action during an episode. We need to remember:
        - How likely was this action? (log_prob for gradient)
        - How good was the immediate outcome? (reward for return calculation)
        
        Args:
            log_prob (torch.Tensor): Log probability of the action taken
                                     Computed by policy.get_action()
                                     This is ∇_θ log π(a|s; θ) waiting to happen
            
            reward (float): Immediate reward received after taking the action
                           For CartPole: +1 if pole still balanced, episode ends otherwise
        """
        # Append to episode storage
        # We'll process these in update() after the episode ends
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        
    def update(self):
        """
        Update policy using REINFORCE after episode completion.
        
        This is where the magic happens! We:
        1. Compute discounted returns G_t for each timestep
        2. Normalize returns (variance reduction)
        3. Compute policy gradient loss
        4. Backpropagate and update weights
        
        Returns:
            float: The loss value (for logging/debugging)
                   Lower isn't necessarily better in RL!
                   The loss is just a surrogate for the gradient.
        
        Mathematical Derivation:
        ------------------------
        We want to maximize:
            J(θ) = E[ Σ_t log π(a_t|s_t; θ) * G_t ]
        
        PyTorch does gradient DESCENT (minimization), so we minimize:
            loss = -Σ_t log π(a_t|s_t; θ) * G_t
        
        The gradient of this loss w.r.t. θ is:
            ∇_θ loss = -Σ_t ∇_θ log π(a_t|s_t; θ) * G_t
        
        When we call loss.backward(), PyTorch computes this gradient.
        When we call optimizer.step(), it does: θ ← θ - lr * ∇_θ loss
        
        Which is equivalent to: θ ← θ + lr * Σ_t ∇_θ log π(a_t|s_t) * G_t
        (the policy gradient update!)
        """
        
        # ======================================================================
        # STEP 1: Calculate discounted returns
        # ======================================================================
        # For each timestep t, compute:
        #     G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T
        #
        # We compute this BACKWARDS for efficiency:
        #     G_T = r_T                    (last step: no future)
        #     G_{T-1} = r_{T-1} + γ*G_T    (one step back)
        #     G_{T-2} = r_{T-2} + γ*G_{T-1}
        #     ...
        #     G_0 = r_0 + γ*G_1
        #
        # This is O(T) instead of O(T²) for computing each G_t from scratch.
        
        returns = []  # Will hold G_t for each timestep
        G = 0         # Running return (starts at 0 = no future after episode ends)
        
        # Iterate through rewards in REVERSE order (from last to first)
        # reversed() creates a reverse iterator without copying the list
        for r in reversed(self.rewards):
            # Recursive formula: G_t = r_t + γ * G_{t+1}
            # G (before update) is G_{t+1}
            # After update, G is G_t
            G = r + self.gamma * G
            
            # Insert at beginning of list (since we're going backwards)
            # After loop: returns[0] = G_0, returns[1] = G_1, etc.
            returns.insert(0, G)
        
        # Convert to PyTorch tensor for vectorized operations
        returns = torch.FloatTensor(returns)
        
        # ======================================================================
        # STEP 2: Normalize returns (variance reduction)
        # ======================================================================
        # Problem: Raw returns can be all positive (e.g., 50, 100, 150)
        # This means ALL actions get reinforced, just by different amounts.
        #
        # Solution: Normalize to mean=0, std=1
        # After normalization:
        #   - Actions with above-average returns get POSITIVE weight (reinforced)
        #   - Actions with below-average returns get NEGATIVE weight (discouraged)
        #
        # This is a simple form of "baseline subtraction" that dramatically
        # reduces variance and speeds up learning.
        
        if len(returns) > 1:
            # Standard score: (x - mean) / std
            # + 1e-8 prevents division by zero if all returns are identical
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ======================================================================
        # STEP 3: Calculate policy gradient loss
        # ======================================================================
        # loss = -Σ_t log π(a_t|s_t) * G_t
        #
        # The negative sign is because:
        # - We want to MAXIMIZE expected return
        # - PyTorch MINIMIZES loss
        # - Minimizing -x is the same as maximizing x
        
        policy_loss = []  # Individual loss terms for each timestep
        
        # zip pairs up corresponding elements: (log_prob_0, G_0), (log_prob_1, G_1), ...
        for log_prob, G in zip(self.log_probs, returns):
            # For each timestep:
            # -log_prob * G
            #
            # If G > 0 (good outcome):
            #   Minimizing -log_prob * G means MAXIMIZING log_prob
            #   → Make this action MORE likely in this state
            #
            # If G < 0 (bad outcome):
            #   Minimizing -log_prob * G means MINIMIZING log_prob
            #   → Make this action LESS likely in this state
            policy_loss.append(-log_prob * G)
        
        # ======================================================================
        # STEP 4: Backpropagate and update
        # ======================================================================
        
        # Clear any existing gradients from previous updates
        # PyTorch ACCUMULATES gradients by default, so we must reset
        self.optimizer.zero_grad()
        
        # Combine all timestep losses into single scalar
        # torch.stack() converts list of tensors to single tensor
        # .sum() adds them all together
        loss = torch.stack(policy_loss).sum()
        
        # Backpropagation: compute gradients
        # This computes ∂loss/∂θ for every parameter θ in the network
        # Gradients are stored in each parameter's .grad attribute
        loss.backward()
        
        # Apply gradients to update parameters
        # For Adam: θ ← θ - lr * m_t / (√v_t + ε)
        # where m_t and v_t are running averages of gradient and squared gradient
        self.optimizer.step()
        
        # ======================================================================
        # STEP 5: Clear episode storage for next episode
        # ======================================================================
        # REINFORCE is episodic: we process one episode at a time
        # Clear the buffers so next episode starts fresh
        self.log_probs = []
        self.rewards = []
        
        # Return loss value for logging (not really meaningful in RL,
        # but useful to check that training is proceeding)
        return loss.item()  # .item() converts 0-d tensor to Python float


# ==============================================================================
#                           TRAINING FUNCTION
# ==============================================================================

def train_cartpole(episodes=500, render=False):
    """
    Train a CartPole policy using the REINFORCE algorithm.
    
    This is the main training loop that:
    1. Creates the environment and policy network
    2. Runs episodes, collecting experience
    3. Updates the policy after each episode
    4. Monitors progress and declares success when "solved"
    
    WHAT DOES "TRAINING" MEAN IN RL?
    ---------------------------------
    Unlike supervised learning (where we have labeled examples), in RL:
    - The agent learns by TRIAL AND ERROR
    - No teacher says "push left here" - the agent discovers this
    - The only feedback is the reward signal (+1 per timestep in CartPole)
    - Training = thousands of attempts to balance the pole
    
    Over time, the policy learns:
    - When pole tilts right → push right to correct
    - When pole tilts left → push left to correct
    - Anticipate and prevent tilting before it gets bad
    
    THE TRAINING LOOP:
    ------------------
    For each episode:
        1. Reset environment (pole starts near vertical)
        2. Loop until pole falls or 500 steps:
            a. Observe current state (position, velocity, angle, angular vel)
            b. Policy outputs action probabilities
            c. SAMPLE action (stochastic for exploration)
            d. Execute action, receive reward, get next state
            e. Store log_prob and reward
        3. Episode ends → compute returns and update policy
        4. Repeat for many episodes
    
    SUCCESS CRITERION:
    ------------------
    CartPole-v1 is considered "solved" when:
        Average reward ≥ 195 over 100 consecutive episodes
    
    Maximum possible reward per episode is 500 (balance for 500 timesteps).
    
    Args:
        episodes (int): Number of training episodes (default 500)
                        More episodes = better policy (usually)
                        Typical: 300-1000 for CartPole
        
        render (bool): Whether to visualize the environment
                       True: Shows window with cart and pole animation
                       False: Runs faster, no visualization
                       Use render=True for debugging/demo
    
    Returns:
        PolicyNetwork: The trained policy, or None if dependencies missing
    """
    
    # --- Check dependencies ---
    # Can't train without PyTorch and Gymnasium!
    if not TORCH_AVAILABLE or not GYM_AVAILABLE:
        print("❌ Cannot train: missing dependencies")
        return None
    
    # ==========================================================================
    # STEP 1: Create the environment
    # ==========================================================================
    # gym.make() creates an environment instance
    # 'CartPole-v1' is the environment identifier
    # - v1 has max 500 steps per episode (v0 has max 200)
    # - Episode ends early if pole falls or cart goes off track
    #
    # render_mode:
    # - 'human': Opens a window showing the cart and pole
    # - None: No visualization (much faster!)
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # ==========================================================================
    # STEP 2: Create policy network and trainer
    # ==========================================================================
    # PolicyNetwork: The neural network that will learn to balance the pole
    # - state_dim=4: Input size (cart pos, cart vel, pole angle, pole vel)
    # - action_dim=2: Output size (push left, push right)
    # - hidden_dim=64: Hidden layer size (64 neurons per layer)
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=64)
    
    # REINFORCETrainer: Implements the policy gradient update
    # - lr=0.01: Learning rate (how big each update step is)
    # - gamma=0.99: Discount factor (how much to value future rewards)
    trainer = REINFORCETrainer(policy, lr=0.01, gamma=0.99)
    
    # ==========================================================================
    # STEP 3: Training loop setup
    # ==========================================================================
    # Track rewards for each episode (to compute running average)
    episode_rewards = []
    
    # Flag to indicate if we've solved the environment
    solved = False
    
    # Print training header
    print("🎮 Training CartPole policy...")
    print("=" * 50)
    
    # ==========================================================================
    # STEP 4: Main training loop
    # ==========================================================================
    # Train for the specified number of episodes
    # Each episode = one attempt to balance the pole from scratch
    for episode in range(episodes):
        
        # --- Reset environment for new episode ---
        # env.reset() returns:
        # - state: Initial observation [cart_pos, cart_vel, pole_angle, pole_vel]
        # - info: Additional information (we ignore with _)
        #
        # Initial state is randomized slightly (pole starts nearly vertical
        # but with small random perturbations)
        state, _ = env.reset()
        
        # Track total reward for this episode
        # For CartPole: +1 per timestep while pole is balanced
        # Goal: Get as close to 500 as possible
        episode_reward = 0
        
        # Episode termination flag
        done = False
        
        # --- Episode loop: keep going until pole falls ---
        while not done:
            
            # 1. SELECT ACTION using current policy
            # policy.get_action() does:
            #   - Forward pass: state → logits → softmax → probabilities
            #   - Sample action from Categorical distribution
            #   - Return action and its log probability
            #
            # action: int, either 0 (left) or 1 (right)
            # log_prob: torch.Tensor, log π(action | state)
            action, log_prob = policy.get_action(state)
            
            # 2. EXECUTE ACTION in environment
            # env.step(action) returns:
            #   - next_state: The new observation after taking action
            #   - reward: Immediate reward (+1 for CartPole if still alive)
            #   - terminated: True if episode ended due to failure
            #                 (pole fell or cart off track)
            #   - truncated: True if episode ended due to time limit
            #                (reached 500 steps = success!)
            #   - info: Additional info (we ignore with _)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Episode is done if terminated OR truncated
            # terminated: Failure (pole fell)
            # truncated: Success (survived 500 timesteps!)
            done = terminated or truncated
            
            # 3. STORE OUTCOME for later policy update
            # We need log_prob (for gradient) and reward (for return calculation)
            trainer.store_outcome(log_prob, reward)
            
            # 4. ACCUMULATE REWARD for logging
            episode_reward += reward
            
            # 5. TRANSITION to next state
            state = next_state
        
        # --- End of episode: update policy ---
        # After episode completes, we have:
        # - trainer.log_probs: List of log probabilities for each action
        # - trainer.rewards: List of rewards for each timestep
        #
        # trainer.update() computes returns and performs gradient update
        loss = trainer.update()
        
        # Record episode reward for tracking progress
        episode_rewards.append(episode_reward)
        
        # --- Print progress every 50 episodes ---
        # Printing every episode would be too noisy
        if (episode + 1) % 50 == 0:
            # Compute average reward over last 50 episodes
            avg_reward = np.mean(episode_rewards[-50:])
            
            # Print status: episode number, average reward, loss
            # f-strings with format specifiers:
            # :4d = integer, 4 characters wide
            # :.1f = float, 1 decimal place
            # :.4f = float, 4 decimal places
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:.1f} | Loss: {loss:.4f}")
            
            # --- Check if environment is "solved" ---
            # CartPole-v1 is solved when average reward ≥ 195 over 100 episodes
            if len(episode_rewards) >= 100:
                avg_last_100 = np.mean(episode_rewards[-100:])
                if avg_last_100 >= 195:
                    print(f"\n🎉 Solved at episode {episode + 1}!")
                    solved = True
                    break  # Stop training early - we've succeeded!
    
    # ==========================================================================
    # STEP 5: Cleanup and report
    # ==========================================================================
    # Close the environment (releases resources, closes window if rendering)
    env.close()
    
    # Compute and print final average reward
    # Use last 100 episodes if available, otherwise use all episodes
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    
    print("=" * 50)
    print(f"Training complete. Final avg reward: {final_avg:.1f}")
    
    # Return the trained policy
    # This can be used for testing or ONNX export
    return policy


# ==============================================================================
#                           ONNX EXPORT FUNCTION
# ==============================================================================

def export_to_onnx(policy, output_path="models/cartpole_policy.onnx"):
    """
    Export trained PyTorch policy to ONNX format.
    
    WHAT IS ONNX?
    -------------
    ONNX (Open Neural Network Exchange) is an open format for representing
    machine learning models. Think of it like PDF for documents - a standardized
    way to share models between different tools and frameworks.
    
    Why export to ONNX?
    - PyTorch model can only run in Python with PyTorch installed
    - ONNX model can run in C++, C#, Java, JavaScript, etc.
    - ONNX Runtime provides optimized inference (often faster than PyTorch)
    - Deploy to edge devices, mobile, web browsers
    - In our project: C++ InferenceEngine loads the ONNX model
    
    HOW DOES ONNX EXPORT WORK?
    --------------------------
    PyTorch uses "tracing" to export models:
    1. We provide a dummy input tensor
    2. PyTorch runs the model and records all operations
    3. The recorded operations become the ONNX graph
    4. Graph is saved to file
    
    Limitations:
    - Can't trace Python control flow (if statements based on tensor values)
    - Dynamic shapes need explicit handling
    - Some PyTorch ops don't have ONNX equivalents
    
    THE ONNX GRAPH:
    ---------------
    Our CartPole policy becomes this ONNX graph:
    
        Input: "state" [batch_size, 4]
           │
           ▼
        MatMul (fc1.weight)
           │
           ▼
        Add (fc1.bias)
           │
           ▼
        Relu
           │
           ▼
        MatMul (fc2.weight)
           │
           ▼
        Add (fc2.bias)
           │
           ▼
        Relu
           │
           ▼
        MatMul (fc3.weight)
           │
           ▼
        Add (fc3.bias)
           │
           ▼
        Output: "action_logits" [batch_size, 2]
    
    Note: Softmax is applied in C++ after loading the ONNX model!
    
    Args:
        policy (PolicyNetwork): Trained PyTorch policy network
                                Must be in eval() mode for deterministic behavior
        
        output_path (str): Path to save the ONNX file
                           Default: "models/cartpole_policy.onnx"
    
    Returns:
        bool: True if export successful, False otherwise
    """
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print("❌ Cannot export: PyTorch not available")
        return False
    
    # ==========================================================================
    # STEP 1: Prepare for export
    # ==========================================================================
    
    # Ensure the output directory exists
    # os.makedirs with exist_ok=True won't error if directory already exists
    # os.path.dirname extracts directory part: "models/file.onnx" → "models"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set model to evaluation mode
    # policy.eval() does several things:
    # 1. Disables dropout (not used here, but common)
    # 2. Disables batch normalization updates (not used here)
    # 3. Signals that we're doing inference, not training
    # ALWAYS call .eval() before exporting!
    policy.eval()
    
    # ==========================================================================
    # STEP 2: Create dummy input for tracing
    # ==========================================================================
    # torch.onnx.export uses "tracing": it runs the model with dummy input
    # and records the operations to build the ONNX graph.
    #
    # The dummy input must have the correct shape and dtype!
    # For CartPole:
    # - Shape: (1, 4) = batch size 1, state dimension 4
    # - dtype: float32 (default for torch.randn)
    #
    # torch.randn generates random values from standard normal distribution
    # The actual values don't matter - only the shape is used
    dummy_input = torch.randn(1, 4)  # [batch_size=1, state_dim=4]
    
    # ==========================================================================
    # STEP 3: Export to ONNX
    # ==========================================================================
    # torch.onnx.export() performs the actual export
    #
    # Parameters explained:
    torch.onnx.export(
        # The PyTorch model to export
        policy,
        
        # Dummy input for tracing
        # Shape defines the expected input shape
        dummy_input,
        
        # Output file path
        # Will create the ONNX file at this location
        output_path,
        
        # export_params=True: Include trained weights in the ONNX file
        # If False, only the graph structure would be exported
        # We want the trained weights so we can do inference!
        export_params=True,
        
        # opset_version: ONNX operator set version
        # Higher versions support more operators but may not work
        # with older ONNX runtimes
        # Version 11 is widely supported and has all ops we need
        opset_version=11,
        
        # do_constant_folding=True: Optimize by pre-computing constant expressions
        # Example: If two constants are multiplied, compute the result now
        # instead of at runtime
        do_constant_folding=True,
        
        # input_names: Names for the input tensors in the ONNX graph
        # These are important for:
        # 1. Documentation - makes the model self-describing
        # 2. C++ code - we reference "state" when feeding input
        input_names=['state'],
        
        # output_names: Names for output tensors
        # We output action_logits (before softmax)
        # C++ code will apply softmax to get probabilities
        output_names=['action_logits'],
        
        # dynamic_axes: Specify which axes can have variable size
        # {0: 'batch_size'} means dimension 0 can vary
        # This allows batched inference with different batch sizes
        # Without this, the model would only accept batch_size=1
        dynamic_axes={
            'state': {0: 'batch_size'},        # [batch, 4] - batch can vary
            'action_logits': {0: 'batch_size'}  # [batch, 2] - batch can vary
        }
    )
    
    print(f"✅ Exported policy to: {output_path}")
    
    # ==========================================================================
    # STEP 4: Validate the exported model
    # ==========================================================================
    # Good practice: Check that the ONNX file is valid after export
    # The onnx package provides validation tools
    
    try:
        # Import the onnx package for validation
        import onnx  # ONNX model manipulation library
        
        # Load the model we just saved
        # onnx.load() parses the protobuf file into a ModelProto object
        model = onnx.load(output_path)
        
        # Check model validity
        # onnx.checker.check_model() verifies:
        # - All nodes have valid inputs/outputs
        # - All operators are defined in the opset
        # - Tensor shapes are consistent
        # Raises an exception if validation fails
        onnx.checker.check_model(model)
        print("✅ ONNX model validation passed")
        
        # Print model info for debugging
        # This helps verify the model has the expected structure
        #
        # model.graph.input[0] is the first input tensor
        # .name gives "state"
        # .type.tensor_type.shape.dim gives dimension info
        # We extract the dimension values (e.g., [1, 4])
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        
        print(f"   Input: {model.graph.input[0].name} {input_shape}")
        print(f"   Output: {model.graph.output[0].name} {output_shape}")
        
    except ImportError:
        # onnx package not installed - that's okay, export still worked
        print("⚠️ ONNX not installed, skipping validation")
        
    except Exception as e:
        # Something went wrong during validation
        # Model might still work, but be cautious
        print(f"⚠️ ONNX validation warning: {e}")
    
    return True


# ==============================================================================
#                           TESTING FUNCTION
# ==============================================================================

def test_policy(policy, episodes=10, render=True):
    """
    Test a trained policy by running episodes deterministically.
    
    TRAINING vs TESTING BEHAVIOR:
    ------------------------------
    During TRAINING:
    - We SAMPLE actions stochastically (exploration)
    - Even if P(left) = 0.99, we might still go right 1% of time
    - This exploration helps discover better strategies
    
    During TESTING:
    - We take the BEST action deterministically (exploitation)
    - Always pick action with highest probability
    - This shows the policy's true learned behavior
    - torch.argmax(probs) instead of sampling
    
    WHY TEST SEPARATELY?
    --------------------
    - Training rewards are noisy (exploration adds randomness)
    - Testing shows actual performance without exploration noise
    - Good sanity check before deployment
    - Fun to watch with render=True!
    
    Args:
        policy (PolicyNetwork): Trained policy to test
        
        episodes (int): Number of test episodes to run
                        More episodes = more reliable average
        
        render (bool): Whether to visualize
                       True: Opens window showing cart and pole
                       False: Runs silently (faster)
    """
    
    # Check dependencies
    if not TORCH_AVAILABLE or not GYM_AVAILABLE:
        print("❌ Cannot test: missing dependencies")
        return
    
    # Create environment for testing
    # render_mode='human' opens visualization window
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # Set policy to evaluation mode
    # During testing:
    # - Dropout is disabled (we don't use it, but good practice)
    # - Batch normalization uses running statistics
    # - Signals "inference mode" to PyTorch
    policy.eval()
    
    print(f"\n🎮 Testing policy for {episodes} episodes...")
    
    # Track rewards across test episodes
    total_rewards = []
    
    # Run test episodes
    for episode in range(episodes):
        
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            
            # --- DETERMINISTIC action selection ---
            # Unlike training, we DON'T sample - we pick the BEST action
            #
            # torch.no_grad() disables gradient computation because:
            # 1. We're not training, so no gradients needed
            # 2. Saves memory and computation
            # 3. Required inside @torch.no_grad() context or with torch.no_grad()
            with torch.no_grad():
                # Convert state to tensor with batch dimension
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Forward pass: get action logits
                logits = policy(state_tensor)
                
                # Convert to probabilities (for interpretability)
                probs = F.softmax(logits, dim=-1)
                
                # Take action with HIGHEST probability (greedy/deterministic)
                # torch.argmax returns index of maximum value
                # .item() converts 0-d tensor to Python int
                action = torch.argmax(probs, dim=-1).item()
            
            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # Record and print episode result
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Reward = {episode_reward}")
    
    # Cleanup
    env.close()
    
    # Print summary statistics
    print(f"\nAverage reward: {np.mean(total_rewards):.1f}")


# ==============================================================================
#                              MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main entry point - parse arguments and run training/export/test.
    
    COMMAND LINE INTERFACE:
    -----------------------
    This script uses argparse for a clean command-line interface.
    
    Examples:
        # Basic training (500 episodes, no export)
        python train_cartpole.py
        
        # Train for more episodes
        python train_cartpole.py --episodes 1000
        
        # Train and export to ONNX
        python train_cartpole.py --export
        
        # Full pipeline: train, export, and test
        python train_cartpole.py --episodes 500 --export --test
        
        # With visualization (slower but fun to watch!)
        python train_cartpole.py --render --test
        
        # Custom output path
        python train_cartpole.py --export --output saved_models/my_policy.onnx
    
    ARGUMENT PARSING:
    -----------------
    argparse provides:
    - Automatic help message (--help)
    - Type checking and conversion
    - Default values
    - Boolean flags (--export, --test, --render)
    - Named arguments (--episodes 500)
    """
    
    # ==========================================================================
    # STEP 1: Set up argument parser
    # ==========================================================================
    # argparse.ArgumentParser creates a parser object
    # description is shown in --help output
    parser = argparse.ArgumentParser(
        description='Train CartPole policy and export to ONNX'
    )
    
    # --- Add command-line arguments ---
    
    # --episodes: Number of training episodes
    # type=int: Automatically convert string input to integer
    # default=500: Use 500 if not specified
    # help: Description shown in --help
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Training episodes (default: 500)'
    )
    
    # --export: Flag to export model to ONNX after training
    # action='store_true': If flag present, args.export = True
    #                      If flag absent, args.export = False
    # This is a boolean flag (no value needed)
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export to ONNX after training'
    )
    
    # --test: Flag to run test episodes after training
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test policy after training'
    )
    
    # --render: Flag to visualize environment
    # Useful for debugging or demonstrations
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment (visualize cart and pole)'
    )
    
    # --output: Custom path for ONNX output file
    # type=str: Keep as string (it's a file path)
    parser.add_argument(
        '--output',
        type=str,
        default='models/cartpole_policy.onnx',
        help='ONNX output path (default: models/cartpole_policy.onnx)'
    )
    
    # ==========================================================================
    # STEP 2: Parse arguments
    # ==========================================================================
    # parser.parse_args() reads sys.argv and returns a Namespace object
    # Access arguments as attributes: args.episodes, args.export, etc.
    args = parser.parse_args()
    
    # ==========================================================================
    # STEP 3: Print configuration
    # ==========================================================================
    # Nice header showing what we're about to do
    print("=" * 60)
    print("  CartPole Policy Training - REINFORCE Algorithm")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Export: {args.export}")
    print(f"  Output: {args.output}")
    print("=" * 60)
    
    # ==========================================================================
    # STEP 4: Train policy
    # ==========================================================================
    # train_cartpole() returns the trained policy, or None if it failed
    policy = train_cartpole(episodes=args.episodes, render=args.render)
    
    # Check for training failure (missing dependencies, etc.)
    if policy is None:
        print("❌ Training failed")
        return  # Exit early
    
    # ==========================================================================
    # STEP 5: Export to ONNX (if requested)
    # ==========================================================================
    # Only export if user specified --export flag
    if args.export:
        export_to_onnx(policy, args.output)
    
    # ==========================================================================
    # STEP 6: Test policy (if requested)
    # ==========================================================================
    # Only test if user specified --test flag
    if args.test:
        test_policy(policy, episodes=5, render=args.render)
    
    # ==========================================================================
    # STEP 7: Done!
    # ==========================================================================
    print("\n✅ Done!")


# ==============================================================================
#                           SCRIPT ENTRY POINT
# ==============================================================================
# This is a Python idiom for "only run main() if this file is executed directly"
#
# When you run: python train_cartpole.py
#   → __name__ is set to "__main__"
#   → main() is called
#
# When you import: from train_cartpole import PolicyNetwork
#   → __name__ is "train_cartpole" (the module name)
#   → main() is NOT called (we just want to import classes)
#
# This allows the file to be both:
# 1. A runnable script (python train_cartpole.py)
# 2. A module for importing (from train_cartpole import PolicyNetwork)

if __name__ == "__main__":
    main()

# ==============================================================================
#                              END OF FILE
# ==============================================================================
#
# LEARNING SUMMARY - KEY CONCEPTS:
# ================================
#
# 1. REINFORCEMENT LEARNING BASICS:
#    - Agent learns by interacting with environment
#    - Actions affect state, state determines reward
#    - Goal: maximize cumulative reward
#
# 2. POLICY GRADIENT (REINFORCE):
#    - Policy π(a|s) maps states to action probabilities
#    - Gradient: ∇J = E[∇log π(a|s) * G]
#    - G (return) = discounted sum of future rewards
#    - Update makes good actions more likely
#
# 3. NEURAL NETWORKS:
#    - Universal function approximators
#    - Learn complex mappings (state → action probs)
#    - Training via backpropagation
#
# 4. PYTORCH ESSENTIALS:
#    - nn.Module: Base class for neural networks
#    - forward(): Define computation
#    - Autograd: Automatic differentiation
#    - Optimizer: Gradient descent variants
#
# 5. ONNX:
#    - Open format for neural networks
#    - Export from PyTorch, run in C++
#    - Enables cross-platform deployment
#
# NEXT STEPS FOR LEARNING:
# ========================
# - A2C (Actor-Critic): Add value function baseline
# - PPO (Proximal Policy Optimization): Clip updates for stability
# - DQN (Deep Q-Network): Value-based approach
# - More complex environments: Atari, MuJoCo, etc.
#
# ==============================================================================
