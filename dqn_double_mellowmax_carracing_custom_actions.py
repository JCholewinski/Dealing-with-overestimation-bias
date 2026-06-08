import random
import re
from collections import deque, namedtuple

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


ENV_NAME = "CarRacing-v3"


CUSTOM_ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),      # 0: do nothing
    np.array([0.0, 1.0, 0.0], dtype=np.float32),      # 1: gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),      # 2: brake

    np.array([-0.4, 0.8, 0.0], dtype=np.float32),     # 3: left + gas
    np.array([0.4, 0.8, 0.0], dtype=np.float32),      # 4: right + gas

    np.array([-0.8, 0.6, 0.0], dtype=np.float32),     # 5: strong left + gas
    np.array([0.8, 0.6, 0.0], dtype=np.float32),      # 6: strong right + gas

    np.array([-1.0, 0.0, 0.0], dtype=np.float32),     # 7: left only
    np.array([1.0, 0.0, 0.0], dtype=np.float32),      # 8: right only
]


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


class DiscreteCarRacingActionWrapper(gym.ActionWrapper):
    """
    DQN selects a discrete action index.
    The wrapper maps it to a continuous CarRacing action:
    [steering, gas, brake].
    """

    def __init__(self, env):
        super().__init__(env)
        self.actions = CUSTOM_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[int(action)]


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeats one selected action for several environment steps.
    This makes control smoother and reduces the number of agent decisions.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        last_obs = None
        terminated = False
        truncated = False
        last_info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            last_obs = obs
            last_info = info

            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, last_info


class FramePreprocessWrapper(gym.ObservationWrapper):
    """
    Converts RGB frame 96x96x3 to grayscale 84x84.
    Output shape: 1x84x84, values in [0, 1].
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 84, 84),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = obs.astype(np.float32) / 255.0

        gray = (
            0.299 * obs[:, :, 0]
            + 0.587 * obs[:, :, 1]
            + 0.114 * obs[:, :, 2]
        )

        gray = gray[:84, :]
        gray = gray[:, 6:90]

        gray = np.expand_dims(gray, axis=0)

        return gray.astype(np.float32)


class FrameStackWrapper(gym.Wrapper):
    """
    Stacks last k grayscale frames.
    Output shape: k x 84 x 84.
    """

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        c, h, w = env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k * c, h, w),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for _ in range(self.k):
            self.frames.append(obs)

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


class RewardClipWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.clip(reward, -1.0, 1.0))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class CnnQNetwork(nn.Module):
    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out_size = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def make_env(
    env_name=ENV_NAME,
    render_mode=None,
    clip_rewards=True,
    action_repeat=4,
):
    """
    CarRacing-v3 uses continuous actions internally.
    We wrap it with our own discrete action set.
    """

    if render_mode is None:
        env = gym.make(env_name, continuous=True)
    else:
        env = gym.make(env_name, continuous=True, render_mode=render_mode)

    env = DiscreteCarRacingActionWrapper(env)
    env = ActionRepeatWrapper(env, repeat=action_repeat)
    env = FramePreprocessWrapper(env)
    env = FrameStackWrapper(env, k=4)

    if clip_rewards:
        env = RewardClipWrapper(env)

    return env


def make_raw_video_env(env_name=ENV_NAME):
    env = gym.make(env_name, continuous=True, render_mode="rgb_array")
    return env


def preprocess_single_frame(obs):
    obs = obs.astype(np.float32) / 255.0

    gray = (
        0.299 * obs[:, :, 0]
        + 0.587 * obs[:, :, 1]
        + 0.114 * obs[:, :, 2]
    )

    gray = gray[:84, :]
    gray = gray[:, 6:90]
    gray = np.expand_dims(gray, axis=0)

    return gray.astype(np.float32)


def slugify(text):
    text = text.lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(x, window=10):
    x = np.asarray(x)

    if len(x) < window:
        return x, np.arange(len(x))

    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    xs = np.arange(window - 1, len(x))

    return ma, xs


def mellowmax(q_values, omega=5.0, dim=1):
    max_q, _ = torch.max(q_values, dim=dim, keepdim=True)
    shifted = q_values - max_q

    mm = max_q.squeeze(dim) + (
        torch.log(torch.mean(torch.exp(omega * shifted), dim=dim)) / omega
    )

    return mm


def select_action(policy_net, state, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        q_values = policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device,
    backup_type: str,
    mellowmax_omega: float = 5.0,
):
    if len(replay_buffer) < batch_size:
        return None

    transitions = replay_buffer.sample(batch_size)

    states = torch.tensor(
        np.array(transitions.state),
        dtype=torch.float32,
        device=device,
    )

    actions = torch.tensor(
        transitions.action,
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)

    rewards = torch.tensor(
        transitions.reward,
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.tensor(
        np.array(transitions.next_state),
        dtype=torch.float32,
        device=device,
    )

    dones = torch.tensor(
        transitions.done,
        dtype=torch.float32,
        device=device,
    )

    current_q = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        if backup_type == "dqn":
            next_q = target_net(next_states).max(dim=1).values

        elif backup_type == "double_dqn":
            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states).gather(1, next_actions).squeeze(1)

        elif backup_type == "mellowmax":
            next_q_values = target_net(next_states)
            next_q = mellowmax(
                next_q_values,
                omega=mellowmax_omega,
                dim=1,
            )

        else:
            raise ValueError(f"Unknown backup_type: {backup_type}")

        target_q = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def evaluate_policy(
    policy_net,
    env_name,
    seed,
    episodes,
    device,
    action_repeat=4,
):
    env = make_env(
        env_name=env_name,
        render_mode=None,
        clip_rewards=False,
        action_repeat=action_repeat,
    )

    returns = []
    lengths = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        total_return = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                action = int(policy_net(state_tensor).argmax(dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            state = next_state
            total_return += reward
            steps += 1

        returns.append(total_return)
        lengths.append(steps)

    env.close()

    return float(np.mean(returns)), float(np.mean(lengths))


def train_agent(
    agent_name: str,
    backup_type: str,
    seed: int = 0,
    env_name: str = ENV_NAME,
    episodes: int = 80,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    buffer_size: int = 50_000,
    learning_starts: int = 500,
    train_every: int = 4,
    target_update_every: int = 1000,
    min_epsilon: float = 0.05,
    epsilon_decay: float = 0.97,
    eval_every: int = 20,
    eval_episodes: int = 1,
    mellowmax_omega: float = 5.0,
    action_repeat: int = 4,
):
    set_seed(seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {agent_name} on {device}")

    env = make_env(
        env_name=env_name,
        render_mode=None,
        clip_rewards=True,
        action_repeat=action_repeat,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)

    input_channels = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Custom actions: {action_dim}")
    print(f"Action repeat: {action_repeat}")

    policy_net = CnnQNetwork(input_channels, action_dim).to(device)
    target_net = CnnQNetwork(input_channels, action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = 1.0
    global_step = 0

    train_returns = []
    eval_returns = []
    avg_max_q_values = []
    episode_lengths = []
    eval_lengths = []
    losses = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        total_return = 0.0
        steps = 0
        episode_q_values = []
        episode_losses = []

        while not done:
            action = select_action(
                policy_net=policy_net,
                state=state,
                epsilon=epsilon,
                action_dim=action_dim,
                device=device,
            )

            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                max_q = policy_net(state_tensor).max(dim=1).values.item()
                episode_q_values.append(max_q)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done),
            )

            state = next_state
            total_return += reward
            steps += 1
            global_step += 1

            if global_step > learning_starts and global_step % train_every == 0:
                loss = optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                    backup_type=backup_type,
                    mellowmax_omega=mellowmax_omega,
                )

                if loss is not None:
                    episode_losses.append(loss)

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        train_returns.append(total_return)
        episode_lengths.append(steps)
        avg_max_q_values.append(float(np.mean(episode_q_values)))
        losses.append(float(np.mean(episode_losses)) if episode_losses else np.nan)

        if (episode + 1) % eval_every == 0:
            eval_return, eval_len = evaluate_policy(
                policy_net=policy_net,
                env_name=env_name,
                seed=seed,
                episodes=eval_episodes,
                device=device,
                action_repeat=action_repeat,
            )
        else:
            eval_return = np.nan
            eval_len = np.nan

        eval_returns.append(eval_return)
        eval_lengths.append(eval_len)

        if (episode + 1) % 5 == 0:
            recent_train = np.mean(train_returns[-5:])

            recent_eval_values = [
                x for x in eval_returns[-10:] if not np.isnan(x)
            ]

            recent_eval = (
                np.mean(recent_eval_values)
                if len(recent_eval_values) > 0
                else np.nan
            )

            print(
                f"{agent_name} | seed={seed} | episode={episode + 1:04d} | "
                f"train_return_5={recent_train:8.1f} | "
                f"eval_return={recent_eval:8.1f} | "
                f"epsilon={epsilon:.3f} | "
                f"steps={global_step} | "
                f"buffer={len(replay_buffer)}"
            )

    env.close()

    result = pd.DataFrame({
        "agent": agent_name,
        "backup_type": backup_type,
        "seed": seed,
        "episode": np.arange(1, episodes + 1),
        "train_return": train_returns,
        "eval_return": eval_returns,
        "avg_max_q": avg_max_q_values,
        "episode_length": episode_lengths,
        "eval_length": eval_lengths,
        "loss": losses,
        "mellowmax_omega": mellowmax_omega if backup_type == "mellowmax" else np.nan,
        "action_repeat": action_repeat,
        "num_custom_actions": len(CUSTOM_ACTIONS),
    })

    model_filename = f"{slugify(agent_name)}_seed{seed}_carracing_custom_actions.pt"
    torch.save(policy_net.state_dict(), model_filename)
    print(f"Saved model: {model_filename}")

    return result


def save_agent_video(
    model_path,
    output_path,
    env_name=ENV_NAME,
    max_agent_steps=250,
    fps=30,
    seed=999,
    action_repeat=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_env = make_raw_video_env(env_name=env_name)

    action_dim = len(CUSTOM_ACTIONS)

    policy_net = CnnQNetwork(input_channels=4, action_dim=action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    frames_for_agent = deque(maxlen=4)
    video_frames = []

    obs, _ = raw_env.reset(seed=seed)
    raw_env.action_space.seed(seed)

    processed = preprocess_single_frame(obs)

    for _ in range(4):
        frames_for_agent.append(processed)

    done = False
    total_return = 0.0
    agent_step = 0

    while not done and agent_step < max_agent_steps:
        frame = raw_env.render()
        video_frames.append(frame)

        state = np.concatenate(list(frames_for_agent), axis=0)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            discrete_action = int(policy_net(state_tensor).argmax(dim=1).item())

        continuous_action = CUSTOM_ACTIONS[discrete_action]

        for _ in range(action_repeat):
            next_obs, reward, terminated, truncated, info = raw_env.step(continuous_action)

            total_return += reward
            done = terminated or truncated

            frame = raw_env.render()
            video_frames.append(frame)

            if done:
                break

        processed = preprocess_single_frame(next_obs)
        frames_for_agent.append(processed)

        agent_step += 1

    for _ in range(30):
        frame = raw_env.render()
        video_frames.append(frame)

    raw_env.close()

    imageio.mimsave(output_path, video_frames, fps=fps)

    print(f"Saved video: {output_path}")
    print(f"Video return: {total_return:.2f}")
    print(f"Video agent steps: {agent_step}")


def plot_metric(results, metric, ylabel, filename, only_eval_points=False):
    plt.figure(figsize=(10, 6))

    for agent in results["agent"].unique():
        agent_df = results[results["agent"] == agent].copy()

        if only_eval_points:
            agent_df = agent_df.dropna(subset=[metric])

        grouped = agent_df.groupby("episode")[metric].mean()

        if len(grouped) == 0:
            continue

        ma, xs = moving_average(grouped.values, window=10)

        plt.plot(
            grouped.index.values[xs],
            ma,
            label=agent,
        )

    plt.title("CarRacing-v3 with custom discrete action set")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_results(results: pd.DataFrame):
    plot_metric(
        results,
        metric="train_return",
        ylabel="Training return, smoothed over 10 episodes",
        filename="carracing_custom_train_return.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="eval_return",
        ylabel="Evaluation return, smoothed over 10 eval points",
        filename="carracing_custom_eval_return.png",
        only_eval_points=True,
    )

    plot_metric(
        results,
        metric="avg_max_q",
        ylabel="Average max Q(s, a), smoothed over 10 episodes",
        filename="carracing_custom_avg_max_q.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="episode_length",
        ylabel="Episode length, smoothed over 10 episodes",
        filename="carracing_custom_episode_length.png",
        only_eval_points=False,
    )


def print_summary(results: pd.DataFrame):
    eval_results = results.dropna(subset=["eval_return"])

    summary_train = (
        results
        .groupby("agent")
        .agg(
            final_20_train_return=("train_return", lambda x: np.mean(x.tail(20))),
            final_20_avg_max_q=("avg_max_q", lambda x: np.mean(x.tail(20))),
            final_20_episode_length=("episode_length", lambda x: np.mean(x.tail(20))),
            best_train_return=("train_return", "max"),
        )
        .reset_index()
    )

    summary_eval = (
        eval_results
        .groupby("agent")
        .agg(
            final_5_eval_return=("eval_return", lambda x: np.mean(x.tail(5))),
            best_eval_return=("eval_return", "max"),
            final_5_eval_length=("eval_length", lambda x: np.mean(x.tail(5))),
        )
        .reset_index()
    )

    summary = summary_train.merge(summary_eval, on="agent", how="left")

    print("\nSummary:")
    print(summary.to_string(index=False))

    summary.to_csv("carracing_custom_summary.csv", index=False)


def select_best_model_seed(results: pd.DataFrame, agent_name: str):
    eval_results = results[
        (results["agent"] == agent_name) & (~results["eval_return"].isna())
    ].copy()

    if len(eval_results) == 0:
        print(f"No evaluation results found for {agent_name}. Using seed 0.")
        return 0

    seed_scores = (
        eval_results
        .groupby("seed")
        .tail(3)
        .groupby("seed")["eval_return"]
        .mean()
    )

    best_seed = int(seed_scores.idxmax())
    best_score = float(seed_scores.max())

    print(f"\nBest {agent_name} seed: {best_seed}")
    print(f"Best final eval return: {best_score:.2f}")

    return best_seed


def main():
    episodes = 150
    seeds = [0]

    agents = [
        {
            "agent_name": "DQN",
            "backup_type": "dqn",
        },
        {
            "agent_name": "Double DQN",
            "backup_type": "double_dqn",
        },
        {
            "agent_name": "Mellowmax DQN",
            "backup_type": "mellowmax",
        },
    ]

    all_results = []

    print(f"Environment: {ENV_NAME}")
    print("Using continuous CarRacing internally.")
    print("DQN sees custom discrete actions mapped to [steering, gas, brake].")
    print("Observation: stacked grayscale frames, shape 4x84x84.")
    print("Network: CNN Q-network.")
    print("Training DQN, Double DQN and Mellowmax DQN.\n")

    for i, action in enumerate(CUSTOM_ACTIONS):
        print(f"Action {i}: {action}")

    print()

    for seed in seeds:
        for agent in agents:
            result = train_agent(
                agent_name=agent["agent_name"],
                backup_type=agent["backup_type"],
                seed=seed,
                env_name=ENV_NAME,
                episodes=episodes,
                batch_size=64,
                gamma=0.99,
                lr=1e-4,
                buffer_size=50_000,
                learning_starts=500,
                train_every=4,
                target_update_every=1000,
                epsilon_decay=0.97,
                min_epsilon=0.05,
                eval_every=20,
                eval_episodes=1,
                mellowmax_omega=5.0,
                action_repeat=4,
            )

            all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv(
        "dqn_double_dqn_mellowmax_carracing_custom_actions_results.csv",
        index=False,
    )

    plot_results(results)
    print_summary(results)

    for agent in ["DQN", "Double DQN", "Mellowmax DQN"]:
        best_seed = select_best_model_seed(results, agent)

        model_path = f"{slugify(agent)}_seed{best_seed}_carracing_custom_actions.pt"
        output_video = f"{slugify(agent)}_carracing_custom_actions_demo.mp4"

        save_agent_video(
            model_path=model_path,
            output_path=output_video,
            env_name=ENV_NAME,
            max_agent_steps=250,
            fps=30,
            seed=999,
            action_repeat=4,
        )

    print("\nSaved files:")
    print("- dqn_double_dqn_mellowmax_carracing_custom_actions_results.csv")
    print("- carracing_custom_summary.csv")
    print("- carracing_custom_train_return.png")
    print("- carracing_custom_eval_return.png")
    print("- carracing_custom_avg_max_q.png")
    print("- carracing_custom_episode_length.png")
    print("- dqn_carracing_custom_actions_demo.mp4")
    print("- double_dqn_carracing_custom_actions_demo.mp4")
    print("- mellowmax_dqn_carracing_custom_actions_demo.mp4")


if __name__ == "__main__":
    main()