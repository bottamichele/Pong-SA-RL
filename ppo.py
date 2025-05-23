import gymnasium as gym
import torch as tc
import numpy as np
import datetime
import pong_gym
import os

from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, FrameStackObservation, FlattenObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics as RecordEpisodeStatisticsVec

from pong_gym.wrappers import NormalizeObservationPong, PointReward

from ppo_algorithm import Rollout
from ppo_algorithm.agent import PPOAgent
from ppo_algorithm.neural_net.nn import NNActorCriticDiscrete

from collections import deque

from torch.utils.tensorboard import SummaryWriter

from utils import FrameSkip

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 5000000
N_ENVS = 8
FRAME_SKIP = 1
FRAME_STACK = 1
N_STEPS = 512
LEARNING_RATE = 10**-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 64
N_EPOCHS = 8
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
DEVICE = tc.device("cpu")
RECORD_VIDEO = True
LOGGING = True

# ========================================
# ================= TRAIN ================
# ========================================

def create_env(training_path, idx_env):
    def aux():
        #Create the enviroment.
        if RECORD_VIDEO and idx_env == 0:
            env = gym.make("pong_gym/Pong-v0", render_mode="rgb_array")
        else:
            env = gym.make("pong_gym/Pong-v0")

        #Pong's wrappers.
        env = NormalizeObservationPong(env)
        env = PointReward(env)

        #Frame skip.
        if FRAME_SKIP > 1:
            env = FrameSkip(env, skip=FRAME_SKIP)

        #Frame stack.
        if FRAME_STACK > 1:
            env = FrameStackObservation(env, FRAME_STACK, padding_type="zero")
            env = FlattenObservation(env)

        #Record video wrapper.
        if RECORD_VIDEO and idx_env == 0:
            env = RecordVideo(env, os.path.join(training_path, "video"), episode_trigger=lambda e:e % 50 == 0, name_prefix="pong")

        return env
    
    return aux

def train():
    training_path = os.path.join("./ppo/", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    training_model_path = os.path.join(training_path, "models")

    #Create the vectorized enviroment.
    envs = SyncVectorEnv([create_env(training_path, idx) for idx in range(N_ENVS)])
    envs = RecordEpisodeStatisticsVec(envs, buffer_length=1)

    #Create the agent.
    agent = PPOAgent(NNActorCriticDiscrete(envs.single_observation_space.shape[0], envs.single_action_space.n, 128, 1, 1, 1),
                     Rollout(N_STEPS, N_ENVS, envs.single_observation_space.shape, (), act_dtype=tc.int32, device=DEVICE),
                     LEARNING_RATE,
                     BATCH_SIZE,
                     N_EPOCHS,
                     DEVICE,
                     gamma=GAMMA,
                     gae_coeff=GAE_LAMBDA,
                     clip_range=CLIP_RANGE,
                     value_coeff=VALUE_COEFFICIENT,
                     entr_coeff=ENTROPY_COEFFICIENT)

    #Tensorboard logger.
    if LOGGING:
        summary = SummaryWriter(os.path.join(training_path, "log"))

    #Training phase.
    total_states = 0
    episode = 1
    scores = deque(maxlen=100)
    ep_length = deque(maxlen=100)

    os.makedirs(training_model_path)

    print("Training is started")

    obs, infos = envs.reset()
    done = np.zeros(N_ENVS, dtype=np.int32)
    while total_states <= TARGET_TOTAL_STEPS:
        for _ in range(N_STEPS):
            #Choose action.
            action, value, log_prob = agent.choose_action(tc.Tensor(obs).to(DEVICE))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, infos = envs.step(action.cpu().numpy())

            #Store one step infos into rollout.
            agent.remember(tc.Tensor(obs).to(DEVICE), 
                           action, 
                           log_prob, 
                           tc.Tensor(reward).to(DEVICE), 
                           tc.Tensor(done).to(DEVICE), 
                           value.reshape(-1))

            #Next observation.
            obs = next_obs
            done = np.logical_or(terminated, truncation)
            total_states += 1

            #Print episode infos.
            if "episode" in infos:
                scores.append(infos["episode"]["r"][infos["_episode"]][0])
                ep_length.append(infos["episode"]["l"][infos["_episode"]][0])
                touch = infos["ball_touched"][infos["_episode"]][0]

                print("- Episode {:>3d}: score = {:.1f}; avg. score = {:.2f}; touch = {:>2d}; states = {:>3d}; total state = {:>6d}".format(episode, scores[-1], np.mean(scores), touch, ep_length[-1], total_states))
                episode += 1

            #Save current policy.
            if total_states % 200000 == 0:
                tc.save(agent.model.state_dict(), os.path.join(training_model_path, f"model_{total_states}.pth"))

        #Train step.
        train_infos = agent.train(tc.Tensor(obs).to(DEVICE), tc.Tensor(done).to(DEVICE))

        if LOGGING:
            if len(scores) > 0:
                summary.add_scalar("episode/avg_score", float(np.mean(scores)), total_states)
            if len(ep_length) > 0:
                summary.add_scalar("episode/avg_length", float(np.mean(ep_length)), total_states)
            summary.add_scalar("train/surrogate_loss", train_infos["surrogate_loss"], total_states)
            summary.add_scalar("train/value_loss", train_infos["value_loss"], total_states)
            summary.add_scalar("train/entropy_loss", train_infos["entropy_loss"], total_states)
            summary.add_scalar("train/total_loss", train_infos["total_loss"], total_states)
            summary.add_scalar("train/clip_fraction", train_infos["clip_fraction"], total_states)
            summary.add_scalar("train/approximate_kl_div", train_infos["approx_kl"], total_states)
            summary.add_scalar("train/explained_variance", train_infos["explained_variance"], total_states)

    envs.close()
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
    model = NNActorCriticDiscrete(env.observation_space.shape[0], env.action_space.n, 128, 1, 1, 1).to("cpu")
    model.load_state_dict(tc.load(os.path.join("./models/", model_name)))

    #Test phase.
    for episode in range(1, n_episodes+1):
        obs, info = env.reset()
        done = False

        while not done:
            #Choose action.
            action, value, _, _ = model.action_and_value(tc.Tensor(obs).unsqueeze(0))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, info = env.step(action.cpu().item())
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