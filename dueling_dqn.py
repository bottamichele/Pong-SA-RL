import gymnasium as gym
import torch as tc
import numpy as np
import datetime
import pong_gym
import os

from gymnasium.wrappers import RecordEpisodeStatistics
from pong_gym.wrappers import NormalizeObservationPong, PointReward

from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.agent import DQNAgent
from dqn_algorithms.policy import greedy_policy

from collections import deque

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 3000000
MEMORY_SIZE = 500000
UPDATE_TARGET_STEP = 5000
GAMMA = 0.99
LEARNING_RATE = 10**-4
BATCH_SIZE = 64
EPSILON_INIT = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 3.96 * 10**-6
DEVICE = tc.device("cpu")
USE_POINT_WRAPPER = False

# ========================================
# ================= TRAIN ================
# ========================================

def train():
    training_path = os.path.join("./dueling_dqn/", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    training_model_path = os.path.join(training_path, "models")

    #Create the enviroment.
    env = gym.make("pong_gym/Pong-v0")
    env = NormalizeObservationPong(env)
    if USE_POINT_WRAPPER:
        env = PointReward(env)
    env = RecordEpisodeStatistics(env, buffer_length=1)

    #Create the agent.
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, 256, 2, 1, 1)
    agent = DQNAgent(model,
                     {"type": "uniform_memory", "mem_size": MEMORY_SIZE, "obs_size": env.observation_space.shape[0]},
                     UPDATE_TARGET_STEP,
                     BATCH_SIZE,
                     LEARNING_RATE,
                     GAMMA,
                     EPSILON_INIT,
                     EPSILON_END,
                     EPSILON_DECAY,
                     DEVICE)

    #Training phase.
    total_states = 0
    episode = 1
    scores = deque(maxlen=100)
    tot_rewards = deque(maxlen=100)
    ep_length = deque(maxlen=100)

    os.makedirs(training_model_path)

    print("Training is started")

    while total_states <= TARGET_TOTAL_STEPS:
        episode_done = False
        obs, infos = env.reset()

        while not episode_done:
            #Choose action.
            action = agent.choose_action(tc.Tensor(obs).to(DEVICE))

            #Perform action chosen.
            next_obs, reward, terminated, truncated, infos = env.step(action)
            episode_done = terminated or truncated

            #Store one step infos into rollout.
            agent.remember(tc.Tensor(obs).to(DEVICE), action, reward, tc.Tensor(next_obs).to(DEVICE), episode_done)

            #Train step.
            agent.train()

            #Update target model.
            if total_states % agent.update_target_step == 0:
                agent.update_target_model()

            #Next observation.
            obs = next_obs
            total_states += 1

            #Save current policy.
            if total_states % 200000 == 0:
                tc.save(agent.model.state_dict(), os.path.join(training_model_path, f"model_{total_states}.pth"))
        
        tot_rewards.append(infos["episode"]["r"])
        ep_length.append(infos["episode"]["l"])
        scores.append(infos["agent_score"] - infos["bot_score"])
        
        #Print episode info.
        print("- Episode {:>3d}: score = {:2d} - {:2d}; r = {:.1f}; avg. r = {:.2f}; touch = {:>2d}; states = {:>3d}; total state = {:>6d}; epsilon = {:.2f}".format(episode, infos["agent_score"], infos["bot_score"], tot_rewards[-1], np.mean(tot_rewards), infos["ball_touched"], ep_length[-1], total_states, agent.epsilon))
        
        #Next episode.
        episode += 1

    env.close()
    agent.save_model(training_model_path)

# ========================================
# ================= TEST =================
# ========================================

def test(model_name, n_episodes):
    #Create the enviroment.
    env = gym.make("pong_gym/Pong-v0", render_mode="human")
    env = NormalizeObservationPong(env)
    env = RecordEpisodeStatistics(env, buffer_length=1)

    #Load the model.
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, 256, 2, 1, 1).to("cpu")
    model.load_state_dict(tc.load(os.path.join("./models/", model_name)))

    #Test phase.
    for episode in range(1, n_episodes+1):
        obs, info = env.reset()
        done = False

        while not done:
            #Choose action.
            action = greedy_policy(model, tc.Tensor(obs))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, info = env.step(action)
            done = terminated or truncation

            #Next observation.
            obs = next_obs

        print("- Episode {:>3d}: score = {:2d} - {:2d}; r = {:.1f}; states = {:>2d}".format(episode, info["agent_score"], info["bot_score"], info["episode"]["r"], info["episode"]["l"]))

    env.close()

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    # train()
    test("dueling_dqn.pth", 100)