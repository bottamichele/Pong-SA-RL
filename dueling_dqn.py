import gymnasium as gym
import torch as tc
import numpy as np
import datetime
import pong_gym
import os

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, FrameStackObservation, FlattenObservation

from pong_gym.wrappers import NormalizeObservationPong, PointReward

from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.agent import DQNAgent
from dqn_algorithms.policy import greedy_policy

from collections import deque

from torch.utils.tensorboard import SummaryWriter

from utils import FrameSkip

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 3000000
FRAME_SKIP = 1
FRAME_STACK = 1
MEMORY_SIZE = 500000
UPDATE_TARGET_STEP = 10000
GAMMA = 0.99
LEARNING_RATE = 10**-4
BATCH_SIZE = 64
EPSILON_INIT = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1.98 * 10**-6
DEVICE = tc.device("cuda")
RECORD_VIDEO = True
LOGGING = True
USE_POINT_WRAPPER = False

# ========================================
# ================= TRAIN ================
# ========================================

def train():
    training_path = os.path.join("./dueling_dqn/", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    training_model_path = os.path.join(training_path, "models")

    #Create the enviroment.
    if RECORD_VIDEO:
        env = gym.make("pong_gym/Pong-v0", render_mode="rgb_array")
    else:
        env = gym.make("pong_gym/Pong-v0")
    env = NormalizeObservationPong(env)
    if USE_POINT_WRAPPER:
        env = PointReward(env)
    if FRAME_SKIP > 1:
        env = FrameSkip(env, skip=FRAME_SKIP)
    if FRAME_STACK > 1:
        env = FrameStackObservation(env, FRAME_STACK, padding_type="zero")
        env = FlattenObservation(env)
    if RECORD_VIDEO:
        env = RecordVideo(env, os.path.join(training_path, "video"), episode_trigger=lambda e:e % 25 == 0, name_prefix="pong")
    env = RecordEpisodeStatistics(env, buffer_length=1)

    #Create the agent.
    agent = DQNAgent(DuelingDQN(env.observation_space.shape[0], env.action_space.n, 128, 1, 1, 1),
                     {"type": "proportional_prioritized_memory", "mem_size": MEMORY_SIZE, "obs_size": env.observation_space.shape[0]},
                     UPDATE_TARGET_STEP,
                     BATCH_SIZE,
                     LEARNING_RATE,
                     GAMMA,
                     EPSILON_INIT,
                     EPSILON_END,
                     EPSILON_DECAY,
                     DEVICE)

    #Tensorboard logger.
    if LOGGING:
        summary = SummaryWriter(os.path.join(training_path, "log"))

    #Training phase.
    total_states = 0
    episode = 1
    beta_decay = (1.0 - agent.memory.beta) / TARGET_TOTAL_STEPS
    scores = deque(maxlen=100)
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
            train_infos = agent.train()
            if LOGGING and "loss" in train_infos.keys():
                summary.add_scalar("train/loss", train_infos["loss"], total_states)

            #Update target model.
            if total_states % agent.update_target_step == 0:
                agent.update_target_model

            #Next observation.
            obs = next_obs
            total_states += 1
            agent.memory.beta += beta_decay

            #Save current policy.
            if total_states % 200000 == 0:
                tc.save(agent.model.state_dict(), os.path.join(training_model_path, f"model_{total_states}.pth"))
        
        scores.append(infos["episode"]["r"])
        ep_length.append(infos["episode"]["l"])

        if LOGGING:
            if len(scores) > 0:
                summary.add_scalar("episode/avg_score", float(np.mean(scores)), total_states)
            if len(ep_length) > 0:
                summary.add_scalar("episode/avg_length", float(np.mean(ep_length)), total_states)
        
        #Print episode info.
        print("- Episode {:>3d}: score = {:.1f}; avg. score = {:.2f}; touch = {:>2d}; states = {:>3d}; total state = {:>6d}; epsilon = {:.2f}".format(episode, scores[-1], np.mean(scores), infos["ball_touched"], ep_length[-1], total_states, agent.epsilon))
        
        #Next episode.
        episode += 1

    env.close()
    if LOGGING:
        summary.close() 
    agent.save_model(training_model_path)

# ========================================
# ================= TEST =================
# ========================================

def test(model_name, n_episodes, frame_skip=1, frame_stack=1):
    #Create the enviroment.
    env = gym.make("pong_gym/Pong-v0", render_mode="human")
    env = NormalizeObservationPong(env)
    if frame_skip > 1:
        env = FrameSkip(env, skip=frame_skip)
    if frame_stack > 1:
        env = FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
        env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env, buffer_length=1)

    #Load the model.
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, 128, 1, 1, 1).to("cpu")
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

        print("- Episode {:>3d}: score = {}; frames = {}".format(episode, info["episode"]["r"], info["episode"]["l"]))

    env.close()

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    train()