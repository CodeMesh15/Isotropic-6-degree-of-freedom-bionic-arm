# -*- coding: utf-8 -*-
"""Kopie von model-based-rl-tutorial-Complete.ipynb


Original file is located at
    https://colab.research.google.com/drive/1wa6S1eKBpTHlOAsnIkmqBoWAfjrpCxAU

# model based rl tutorial

## Problem Statement

Implement the Tabular Dyna-Q Algorithm to reach to a goal state

![image.png](https://github.com/NeuromatchAcademy/course-content/blob/main/tutorials/static/W3D4_Tutorial4_QuentinsWorld.png?raw=true)

No. of states = 100  
No. of possible actions = 4 (right, up, left, and down)  

**Rules**:
*   Moving into one of the red states incurs a reward of -1
*   Moving into the world borders stays in the same place
*   Moving into the goal state (yellow square in the upper right corner) gives you a reward of 1
*   Moving anywhere from the goal state ends the episode

## Given

The agent starts in the green state.

## Goal

The goal of the agent is to move from the start (green) location to the goal (yellow) region, while avoiding the red walls.

# Project Pipeline

---
**TABULAR DYNA-Q**

Initialize $Q(s,a)$ and $Model(s,a)$ for all $s \in S$ and $a \in A$.

Loop forever:

> (a) $S$ &larr; current (nonterminal) state <br>
> (b) $A$ &larr; $\epsilon$-greedy$(S,Q)$ <br>
> (c) Take action $A$; observe resultant reward, $R$, and state, $S'$ <br>
> (d) $Q(S,A)$ &larr; $Q(S,A) + \alpha \left[R + \gamma \max_{a} Q(S',a) - Q(S,A)\right]$ <br>
> (e) $Model(S,A)$ &larr; $R,S'$ (assuming deterministic environment) <br>
> (f) Loop repeat $k$ times: <br>
>> $S$ &larr; random previously observed state <br>
>> $A$ &larr; random action previously taken in $S$ <br>
>> $R,S'$ &larr; $Model(S,A)$ <br>
>> $Q(S,A)$ &larr; $Q(S,A) + \alpha \left[R + \gamma \max_{a} Q(S',a) - Q(S,A)\right]$ <br>


---

## **Activity** 1: Environment Initialization

### **Concepts for this activity**

**State**: Represents current environment status.  
**Action**: Agent's available moves.  
**Reward**: Immediate feedback for action taken.  
**Next State**: Follows current state after action.  
**Transition Model**: Function that gives the next state and reward, given the current state and action.

### **A**1.0 Create the QuentinsWorld environment
"""

# Base class of the world (Given)
class world(object):
    def __init__(self):
        return

    def get_outcome(self):
        print("Abstract method, not implemented")
        return

    def get_all_outcomes(self):
        outcomes = {}
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state, reward = self.get_outcome(state, action)
                outcomes[state, action] = [(1, next_state, reward)]

# Derived class - Quentin's World
class QuentinsWorld(world):
    """
    World: Quentin's world.
    100 states (10-by-10 grid world).
    The mapping from state to the grid is as follows:
    90 ...       99
    ...
    40 ...       49
    30 ...       39
    20 21 22 ... 29
    10 11 12 ... 19
    0  1  2  ...  9
    54 is the start state.
    Actions 0, 1, 2, 3 correspond to right, up, left, down.
    Moving anywhere from state 99 (goal state) will end the session.
    Landing in red states incurs a reward of -1.
    Landing in the goal state (99) gets a reward of 1.
    Going towards the border when already at the border will stay in the same
        place.
    """
    def __init__(self):
        self.name = "QuentinsWorld"
        self.n_states = 100
        self.n_actions = 4
        self.dim_x = 10
        self.dim_y = 10
        self.init_state = 54
        self.shortcut_state = 64

    def toggle_shortcut(self):
      if self.shortcut_state == 64:
        self.shortcut_state = 2
      else:
        self.shortcut_state = 64

    def get_outcome(self, state, action):
        if state == 99:  # goal state
            reward = 0
            next_state = None
            return next_state, reward
        reward = 0  # default reward value
        if action == 0:  # move right
            next_state = state + 1
            if state == 98:  # next state is goal state
                reward = 1
            elif state % 10 == 9:  # right border
                next_state = state
            elif state in [11, 21, 31, 41, 51, 61, 71,
                           12, 72,
                           73,
                           14, 74,
                           15, 25, 35, 45, 55, 65, 75]:  # next state is red
                reward = -1
        elif action == 1:  # move up
            next_state = state + 10
            if state == 89:  # next state is goal state
                reward = 1
            if state >= 90:  # top border
                next_state = state
            elif state in [2, 12, 22, 32, 42, 52, 62,
                           3, 63,
                           self.shortcut_state,
                           5, 65,
                           6, 16, 26, 36, 46, 56, 66]:  # next state is red
                reward = -1
        elif action == 2:  # move left
            next_state = state - 1
            if state % 10 == 0:  # left border
                next_state = state
            elif state in [17, 27, 37, 47, 57, 67, 77,
                           16, 76,
                           75,
                           14, 74,
                           13, 23, 33, 43, 53, 63, 73]:  # next state is red
                reward = -1
        elif action == 3:  # move down
            next_state = state - 10
            if state <= 9:  # bottom border
                next_state = state
            elif state in [22, 32, 42, 52, 62, 72, 82,
                           23, 83,
                           84,
                           25, 85,
                           26, 36, 46, 56, 66, 76, 86]:  # next state is red
                reward = -1
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward

"""### **A**1.1 Initialize the Quentin's World environment"""

# Initialize the Quentin's World

env = QuentinsWorld()

"""### **A**1.2 Print the number of states and actions"""

# Print the number of states and actions

print(env.n_states)
print(env.n_actions)

"""### **A**1.3 Import Jax"""

# Import Jax.numpy

import jax.numpy as jp

"""### **A**1.4 Initialise a uniform value function"""

# Initialise a uniform value function

value = jp.ones((env.n_states, env.n_actions))
print(f"{value=}")

"""### **A**1.5 Initialise the transition model"""

# Initialise the transition model

model = jp.nan * jp.zeros((env.n_states, env.n_actions, 2))
print(f"{model=}")

"""### **A**1.6 Define maximum steps for each episode"""

# Define number of episodes and maximum steps for each episode

n_episodes = 500
max_steps = 1000

"""### **Assessment**

* How are states and actions represented for the Quentin's World environment?
* How many Q-Values do we have for each state?
* What are the inputs and outputs of a transition model?

## **Activity** 2: Q-Learning

### **Concepts for this activity**

[Q-Learning](https://compneuro.neuromatch.io/tutorials/W3D4_ReinforcementLearning/student/W3D4_Tutorial3.html#section-2-q-learning)

### **A**2.1 Take random action
"""

# Define random seed

import time
from jax import random
seed = int(time.time() * 1e6) % (2**32)
key = random.PRNGKey(seed)

# Define a reinforcement learning loop by taking random action for a number of steps for some episodes

for episode in range(n_episodes):
    for t in range(max_steps):
        pass

# Initialise the state and reward sum for each episode

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    # Initialise the state
    state = env.init_state
    print(f"{state=}")
    # Initialise the reward sum
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        pass

# Take random action at each step and update the next state and reward

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    # Initialise the state
    state = env.init_state
    print(f"{state=}")
    # Initialise the reward sum
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        print(f"{env.n_actions=}")
        # Take random action
        key, subkey = random.split(key)
        action = random.choice(subkey, env.n_actions)
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print(f"{reward_sum=}")

"""### **A**2.2 Add parameters for Q-Learning"""

# Define parameters for the Q-Learning Algorithm

params = {
    'epsilon': 0.05, # for epsilon-greedy policy
    'alpha': 0.5, # learning_rate
    'gamma': 0.8, # temporal discount factor
    'k': 10, # number of Dyna-Q planning steps
}

"""### **A**2.3 Write a function to update the Q-value"""

# Define the function to update the Value function

def q_learning(state, action, reward, next_state, value, params):
    pass

# Define previous value

def q_learning(state, action, reward, next_state, value, params):
    # Define previous value
    prev_value = value[int(state), int(action)]

# Define max value

def q_learning(state, action, reward, next_state, value, params):
    # Define previous value
    prev_value = value[int(state), int(action)]
    # Define max value
    if next_state is None or jp.isnan(next_state):
        max_value = 0
    else:
        max_value = jp.max(value[int(next_state)])

# Define reward prediction error

def q_learning(state, action, reward, next_state, value, params):
    # Define previous value
    prev_value = value[int(state), int(action)]
    # Define max value
    if next_state is None or jp.isnan(next_state):
        max_value = 0
    else:
        max_value = jp.max(value[int(next_state)])
    # Define reward prediction error
    delta = reward + params['gamma']*max_value - prev_value
    value = value.at[int(state), int(action)].set(prev_value + params['alpha'] * delta)
    return value

"""### **A**2.4 Define an epsilon-greedy policy function

"""

# Define epsilon-greedy function

import jax.random as random
def epsilon_greedy(key, q, epsilon):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    be_greedy = random.uniform(key, ()) > epsilon
    if be_greedy > 0:
        action = jp.argmax(q)
    else:
        action = random.choice(subkey, env.n_actions)
    return action

"""### **A**2.5 Replace random policy with epsilon-greedy policy

"""

# Replace random policy with epsilon-greedy policy

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        print(f"{env.n_actions=}")
        # Take action based on epsilon-greedy policy
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print("{reward_sum=}")

"""### **A**2.6 update value function

"""

# Update value function

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        print(f"{env.n_actions=}")
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print("{reward_sum=}")
        # Update value function
        value = q_learning(state, action, reward, next_state, value, params)

"""### **A**2.7 Update state

"""

# Update state

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        print(f"{env.n_actions=}")
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print("{reward_sum=}")
        value = q_learning(state, action, reward, next_state, value, params)
        # Update state
        if next_state is None:
            break
        state = next_state

"""### **Assessment**

* Explain the Epsilon-Greedy policy.
* Define the reward prediction error.
* Explain the steps for Q-Learning.

## **Activity** 3: Model update and planning

### **Concepts for this activity**

In theory, one can think of a Dyna-Q agent as implementing acting, learning, and planning simultaneously, at all times. But, in practice, one needs to specify the algorithm as a sequence of steps. The most common way in which the Dyna-Q agent is implemented is by adding a planning routine to a Q-learning agent: after the agent acts in the real world and learns from the observed experience, the agent is allowed a series of `k`
 planning steps. At each one of those `k`
 planning steps, the model generates a simulated experience by randomly sampling from the history of all previously experienced state-action pairs. The agent then learns from this simulated experience, again using the same Q-learning rule that you implemented for learning from real experience. This simulated experience is simply a one-step transition, i.e., a state, an action, and the resulting state and reward. So, in practice, a Dyna-Q agent learns (via Q-learning) from one step of real experience during acting, and then from k steps of simulated experience during planning.

### **A**3.1 Update model
"""

# Define model update function

def model_update(model, state, action, reward, next_state):
    model = model.at[state, action, 0].set(reward)
    model = model.at[state, action, 1].set(next_state)
    return model

# Update the model

for episode in range(n_episodes):
    print(f"Episode {episode+1}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t+1}:")
        print(f"{env.n_actions=}")
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum = reward
        print("{reward_sum=}")
        value = q_learning(state, action, reward, next_state, value, params)
        # Update the model
        model = model_update(model, state, action, reward, next_state)
        if next_state is None:
            break
        state = next_state

"""### **A**3.2 Define the planner

"""

# Define the planner function

def planner(key, model, value, params):
    key, subkey = random.split(key)
    for _ in range(params['k']):

        # Find state-action combinations for which we've experienced a reward i.e.
        # the reward value is not NaN. The outcome of this expression is an Nx2
        # matrix, where each row is a state and action value, respectively.
        candidates = jp.array(jp.where(~jp.isnan(model[:, :, 0]))).T

        # Write an expression for selecting a random row index from our candidates
        idx = random.choice(subkey, candidates.shape[0])

        # Obtain the randomly selected state and action values from the candidates
        state, action = candidates[idx]

        # Obtain the expected reward and next state from the model
        reward, next_state = model[state, action]

        # Update the value function using Q-learning
        value = q_learning(state, action, reward, next_state, value, params)
    return value

# Execute planner

# Define loop for number of episodes
for episode in range(n_episodes):
    print(f"Episode {episode}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t}:")
        print(f"{env.n_actions=}")
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print("{reward_sum=}")
        value = q_learning(state, action, reward, next_state, value, params)
        model = model_update(model, state, action, reward, next_state)
        # Execute planner
        value = planner(key, model, value, params)
        if next_state is None:
            break
        state = next_state

"""### **Assessment**

* Explain the model update step of Dyna-Q.
* Explain the planner step of Dyna-Q.
* How many steps of acting and planning does the Dyna-Q agent require for learning?

## **Activity** 4: Visualization

### **A**4.1 Add plotters
"""

# Run learning
reward_sums = jp.zeros(n_episodes)
episode_steps = jp.zeros(n_episodes)

import numpy as np
from scipy.signal import convolve as conv
import matplotlib.pyplot as plt
def plot_state_action_values(env, value, ax=None):
  """
  Generate plot showing value of each action at each state.
  """
  if ax is None:
    fig, ax = plt.subplots()
  for a in range(env.n_actions):
    ax.plot(range(env.n_states), value[:, a], marker='o', linestyle='--')
  ax.set(xlabel='States', ylabel='Values')
  ax.legend(['R','U','L','D'], loc='lower right')
def plot_quiver_max_action(env, value, ax=None):
  """
  Generate plot showing action of maximum value or maximum probability at
    each state (not for n-armed bandit or cheese_world).
  """
  if ax is None:
    fig, ax = plt.subplots()
  X = np.tile(np.arange(env.dim_x), [env.dim_y,1]) + 0.5
  Y = np.tile(np.arange(env.dim_y)[::-1][:,np.newaxis], [1,env.dim_x]) + 0.5
  which_max = np.reshape(value.argmax(axis=1), (env.dim_y,env.dim_x))
  which_max = which_max[::-1,:]
  U = np.zeros(X.shape)
  V = np.zeros(X.shape)
  U[which_max == 0] = 1
  V[which_max == 1] = 1
  U[which_max == 2] = -1
  V[which_max == 3] = -1
  ax.quiver(X, Y, U, V)
  ax.set(
      title='Maximum value/probability actions',
      xlim=[-0.5, env.dim_x+0.5],
      ylim=[-0.5, env.dim_y+0.5],
  )
  ax.set_xticks(np.linspace(0.5, env.dim_x-0.5, num=env.dim_x))
  ax.set_xticklabels(["%d" % x for x in np.arange(env.dim_x)])
  ax.set_xticks(np.arange(env.dim_x1), minor=True)
  ax.set_yticks(np.linspace(0.5, env.dim_y-0.5, num=env.dim_y))
  ax.set_yticklabels(["%d" % y for y in np.arange(0, env.dim_y*env.dim_x, env.dim_x)])
  ax.set_yticks(np.arange(env.dim_y1), minor=True)
  ax.grid(which='minor',linestyle='-')
def plot_heatmap_max_val(env, value, ax=None):
  """
  Generate heatmap showing maximum value at each state
  """
  if ax is None:
    fig, ax = plt.subplots()
  if value.ndim == 1:
      value_max = np.reshape(value, (env.dim_y,env.dim_x))
  else:
      value_max = np.reshape(value.max(axis=1), (env.dim_y,env.dim_x))
  value_max = value_max[::-1,:]
  im = ax.imshow(value_max, aspect='auto', interpolation='none', cmap='afmhot')
  ax.set(title='Maximum value per state')
  ax.set_xticks(np.linspace(0, env.dim_x-1, num=env.dim_x))
  ax.set_xticklabels(["%d" % x for x in np.arange(env.dim_x)])
  ax.set_yticks(np.linspace(0, env.dim_y-1, num=env.dim_y))
  if env.name != 'windy_cliff_grid':
      ax.set_yticklabels(
  ["%d" % y for y in np.arange(
      0, env.dim_y*env.dim_x, env.dim_x)][::-1])
  return im
def plot_rewards(n_episodes, rewards, average_range=10, ax=None):
  """
  Generate plot showing total reward accumulated in each episode.
  """
  if ax is None:
    fig, ax = plt.subplots()
  smoothed_rewards = (conv(rewards, np.ones(average_range), mode='same')
      / average_range)
  ax.plot(range(0, n_episodes, average_range),
  smoothed_rewards[0:n_episodes:average_range],
  marker='o', linestyle='--')
  ax.set(xlabel='Episodes', ylabel='Total reward')
def plot_performance(env, value, n_episodes, reward_sums):
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
  plot_state_action_values(env, value, ax=axes[0,0])
  plot_quiver_max_action(env, value, ax=axes[0,1])
  plot_rewards(n_episodes, reward_sums, ax=axes[1,0])
  im = plot_heatmap_max_val(env, value, ax=axes[1,1])
  fig.colorbar(im)

# Define loop for number of episodes
for episode in range(n_episodes):
    print(f"Episode {episode}:")
    state = env.init_state
    print(f"{state=}")
    reward_sum = 0
    for t in range(max_steps):
        print(f"Step {t}:")
        print(f"{env.n_actions=}")
        action = epsilon_greedy(key, value[state], params['epsilon'])
        print(f"{action=}")
        next_state, reward = env.get_outcome(state, action)
        reward_sum += reward
        print("{reward_sum=}")
        if next_state is None:
            break
        state = next_state
        value = q_learning(state, action, reward, next_state, value, params)
        model = model_update(model, state, action, reward, next_state)
        value = planner(key, model, value, params)
    reward_sums = reward_sums.at[episode].set(reward_sum)
    episode_steps = episode_steps.at[episode].set(t+1)
# Plot performance curves
plot_performance(env,value, n_episodes, reward_sums)

"""# References

[Model-based Reinforcement Learning by Neuromatch](https://compneuro.neuromatch.io/tutorials/W3D4_ReinforcementLearning/student/W3D4_Tutorial4.html)
"""
