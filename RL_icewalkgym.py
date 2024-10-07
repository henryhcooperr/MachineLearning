import os
import random
import numpy as np
import gym
import imageio
from tqdm.notebook import trange
from IPython.display import Image

# Environment setup
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample()) 
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())

state_space = env.observation_space.n
action_space = env.action_space.n

print("There are ", state_space, " possible states")
print("There are ", action_space, " possible actions")

# Q-table initialization
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

Qtable_frozelake = initialize_q_table(state_space, action_space)

# Policy definitions
def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample()
    return action

def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

# Training parameters
n_training_episodes = 10000
learning_rate = 0.7        
gamma = 0.95               # Discount factor
max_steps = 99             
decay_rate = 0.0005        

# Exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.05

# Evaluation parameters
n_eval_episodes = 100
eval_seed = []             

# Training function
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in trange(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state = env.reset()
        done = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, done, info = env.step(action)
            Qtable[state][action] += learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
            
            if done:
                break
            
            state = new_state
    return Qtable

# Train the agent
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozelake)

print(Qtable_frozelake)

# Evaluation function
def evaluate_agent(env, max_steps, n_eval_episodes, Qtable, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset(seed=seed[episode]) if seed else env.reset()
        total_rewards_ep = 0
        
        for step in range(max_steps):
            action = np.argmax(Qtable[state][:])
            state, reward, done, info = env.step(action)
            total_rewards_ep += reward
            if done:
                break
        episode_rewards.append(total_rewards_ep)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

# Evaluate the agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# Video recording function
def record_video(env, Qtable, video_path, fps):
    directory = os.path.dirname(video_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    images = [] 
    done = False
    state = env.reset(seed=random.randint(0, 500))
    images.append(env.render(mode='rgb_array'))
    
    while not done:
        action = np.argmax(Qtable[state][:])
        state, reward, done, info = env.step(action)
        images.append(env.render(mode='rgb_array'))
    
    imageio.mimsave(video_path, [np.array(img) for img in images], duration=200)
    print(f"Video saved to {video_path}")

# Record and display video
video_path = "./replay.gif"
video_fps = 10
record_video(env, Qtable_frozelake, video_path, video_fps)
Image(video_path)
