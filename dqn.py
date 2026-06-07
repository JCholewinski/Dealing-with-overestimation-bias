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


GOAL_HEIGHT = 1.5
ENV_NAME = "Acrobot-v1"


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


class CustomGoalAcrobot(gym.Wrapper):
    """
    Standard Acrobot-v1 terminates when height >= 1.0.
    This wrapper changes the goal height to a custom value, e.g. 1.8.

    Reward remains Acrobot-like:
    -1 for each step until the custom goal is reached,
     0 when the custom goal is reached.
    """

    def __init__(self, env, goal_height=1.8):
        super().__init__(env)
        self.goal_height = goal_height

    def _compute_height(self, state):
        cos_theta1 = state[0]
        sin_theta1 = state[1]
        cos_theta2 = state[2]
        sin_theta2 = state[3]

        cos_theta1_plus_theta2 = (
            cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        )

        height = -cos_theta1 - cos_theta1_plus_theta2

        return float(height)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        info["height"] = self._compute_height(state)
        info["goal_height"] = self.goal_height
        return state, info

    def step(self, action):
        next_state, original_reward, original_terminated, truncated, info = self.env.step(action)

        height = self._compute_height(next_state)
        custom_terminated = height >= self.goal_height

        reward = 0.0 if custom_terminated else -1.0

        info["height"] = height
        info["goal_height"] = self.goal_height
        info["original_reward"] = original_reward
        info["original_terminated"] = original_terminated
        info["custom_terminated"] = custom_terminated

        return next_state, reward, custom_terminated, truncated, info


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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_env(env_name=ENV_NAME, goal_height=GOAL_HEIGHT, render_mode=None):
    if render_mode is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode=render_mode)

    env = CustomGoalAcrobot(env, goal_height=goal_height)
    return env


def slugify(text):
    text = text.lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x, window=10):
    x = np.asarray(x)

    if len(x) < window:
        return x, np.arange(len(x))

    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    xs = np.arange(window - 1, len(x))

    return ma, xs


def mellowmax(q_values, omega=5.0, dim=1):
    """
    mellowmax(x) = 1/omega * log(mean(exp(omega * x)))
    Stable implementation.
    """
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
        device=device
    )

    actions = torch.tensor(
        transitions.action,
        dtype=torch.long,
        device=device
    ).unsqueeze(1)

    rewards = torch.tensor(
        transitions.reward,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        np.array(transitions.next_state),
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        transitions.done,
        dtype=torch.float32,
        device=device
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


def evaluate_policy(policy_net, env_name, seed, episodes, device, goal_height):
    env = make_env(env_name, goal_height=goal_height)
    returns = []
    max_heights = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        total_return = 0.0
        episode_max_height = -999.0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                action = int(policy_net(state_tensor).argmax(dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            state = next_state
            total_return += reward
            episode_max_height = max(episode_max_height, info.get("height", -999.0))

        returns.append(total_return)
        max_heights.append(episode_max_height)

    env.close()

    return float(np.mean(returns)), float(np.mean(max_heights))


def train_agent(
    agent_name: str,
    backup_type: str,
    seed: int = 0,
    env_name: str = ENV_NAME,
    goal_height: float = GOAL_HEIGHT,
    episodes: int = 700,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 5e-4,
    buffer_size: int = 100_000,
    min_epsilon: float = 0.05,
    epsilon_decay: float = 0.997,
    target_update_every: int = 1000,
    eval_every: int = 10,
    eval_episodes: int = 5,
    mellowmax_omega: float = 5.0,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_name, goal_height=goal_height)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = 1.0
    global_step = 0

    train_returns = []
    eval_returns = []
    avg_max_q_values = []
    train_max_heights = []
    eval_max_heights = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        total_return = 0.0
        episode_q_values = []
        episode_max_height = -999.0

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
                    device=device
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
            global_step += 1
            episode_max_height = max(episode_max_height, info.get("height", -999.0))

            optimize_model(
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

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        train_returns.append(total_return)
        avg_max_q_values.append(float(np.mean(episode_q_values)))
        train_max_heights.append(episode_max_height)

        if (episode + 1) % eval_every == 0:
            eval_return, eval_max_height = evaluate_policy(
                policy_net=policy_net,
                env_name=env_name,
                seed=seed,
                episodes=eval_episodes,
                device=device,
                goal_height=goal_height,
            )
        else:
            eval_return = np.nan
            eval_max_height = np.nan

        eval_returns.append(eval_return)
        eval_max_heights.append(eval_max_height)

        if (episode + 1) % 25 == 0:
            recent_train = np.mean(train_returns[-25:])

            recent_eval_values = [
                x for x in eval_returns[-25:] if not np.isnan(x)
            ]

            recent_eval = (
                np.mean(recent_eval_values)
                if len(recent_eval_values) > 0
                else np.nan
            )

            recent_height_values = [
                x for x in eval_max_heights[-25:] if not np.isnan(x)
            ]

            recent_height = (
                np.mean(recent_height_values)
                if len(recent_height_values) > 0
                else np.nan
            )

            print(
                f"{agent_name} | seed={seed} | episode={episode + 1:03d} | "
                f"train_return_25={recent_train:.1f} | "
                f"eval_return={recent_eval:.1f} | "
                f"eval_max_height={recent_height:.2f} | "
                f"epsilon={epsilon:.3f}"
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
        "train_max_height": train_max_heights,
        "eval_max_height": eval_max_heights,
        "goal_height": goal_height,
        "mellowmax_omega": mellowmax_omega if backup_type == "mellowmax" else np.nan,
    })

    model_filename = f"{slugify(agent_name)}_goal{goal_height}_seed{seed}_acrobot.pt"
    torch.save(policy_net.state_dict(), model_filename)
    print(f"Saved model: {model_filename}")

    return result


def draw_goal_line_on_frame(frame, goal_height, color=(255, 0, 0), thickness=4):
    """
    Draw a horizontal custom goal line on Acrobot frame.

    The Acrobot renderer uses a roughly centered coordinate system.
    Link lengths are 1 + 1, so vertical height is approximately in [-2, 2].
    We use a slightly larger world bound for visual alignment.
    """
    frame = frame.copy()

    height_px, width_px, _ = frame.shape

    world_bound = 2.2
    scale = height_px / (2 * world_bound)
    center_y = height_px // 2

    # World y is positive upward, image y is positive downward.
    y_px = int(center_y - goal_height * scale)
    y_px = max(0, min(height_px - 1, y_px))

    y_start = max(0, y_px - thickness // 2)
    y_end = min(height_px, y_px + thickness // 2 + 1)

    frame[y_start:y_end, :, :] = color

    return frame


def save_agent_video(
    model_path,
    output_path,
    env_name=ENV_NAME,
    goal_height=GOAL_HEIGHT,
    max_steps=500,
    fps=30,
    seed=999,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_name, goal_height=goal_height, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    frames = []

    state, _ = env.reset(seed=seed)
    done = False
    total_return = 0.0
    max_height_reached = -999.0
    step = 0

    while not done and step < max_steps:
        frame = env.render()

        frame = draw_goal_line_on_frame(
            frame=frame,
            goal_height=goal_height,
            color=(255, 0, 0),
            thickness=4,
        )

        frames.append(frame)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            action = int(policy_net(state_tensor).argmax(dim=1).item())

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        state = next_state
        total_return += reward
        max_height_reached = max(max_height_reached, info.get("height", -999.0))
        step += 1

    for _ in range(20):
        frame = env.render()

        frame = draw_goal_line_on_frame(
            frame=frame,
            goal_height=goal_height,
            color=(255, 0, 0),
            thickness=4,
        )

        frames.append(frame)

    env.close()

    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Saved video: {output_path}")
    print(f"Video return for {output_path}: {total_return}")
    print(f"Max height reached: {max_height_reached:.3f}")
    print(f"Goal height: {goal_height:.3f}")


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
            label=agent
        )

    plt.title(f"Custom Goal Acrobot-v1, goal_height={GOAL_HEIGHT}")
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
        ylabel="Training return, moving average window=10",
        filename="custom_goal_acrobot_train_return.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="eval_return",
        ylabel="Greedy evaluation return, moving average window=10",
        filename="custom_goal_acrobot_eval_return.png",
        only_eval_points=True,
    )

    plot_metric(
        results,
        metric="avg_max_q",
        ylabel="Average max Q(s, a), moving average window=10",
        filename="custom_goal_acrobot_avg_max_q.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="eval_max_height",
        ylabel="Evaluation max height reached, moving average window=10",
        filename="custom_goal_acrobot_eval_max_height.png",
        only_eval_points=True,
    )


def print_summary(results: pd.DataFrame):
    eval_results = results.dropna(subset=["eval_return"])

    summary_train = (
        results
        .groupby("agent")
        .agg(
            final_50_train_return=("train_return", lambda x: np.mean(x.tail(50))),
            final_50_avg_max_q=("avg_max_q", lambda x: np.mean(x.tail(50))),
            final_50_train_max_height=("train_max_height", lambda x: np.mean(x.tail(50))),
            best_train_return=("train_return", "max"),
        )
        .reset_index()
    )

    summary_eval = (
        eval_results
        .groupby("agent")
        .agg(
            final_10_eval_return=("eval_return", lambda x: np.mean(x.tail(10))),
            final_10_eval_max_height=("eval_max_height", lambda x: np.mean(x.tail(10))),
            best_eval_return=("eval_return", "max"),
            best_eval_max_height=("eval_max_height", "max"),
        )
        .reset_index()
    )

    summary = summary_train.merge(summary_eval, on="agent", how="left")

    print("\nSummary:")
    print(summary.to_string(index=False))

    summary.to_csv("custom_goal_acrobot_summary.csv", index=False)


def select_best_model_seed(results: pd.DataFrame, agent_name: str):
    eval_results = results[
        (results["agent"] == agent_name) & (~results["eval_return"].isna())
    ].copy()

    if len(eval_results) == 0:
        raise ValueError(f"No evaluation results found for agent: {agent_name}")

    seed_scores = (
        eval_results
        .groupby("seed")
        .tail(5)
        .groupby("seed")["eval_return"]
        .mean()
    )

    best_seed = int(seed_scores.idxmax())
    best_score = float(seed_scores.max())

    print(f"\nBest {agent_name} seed: {best_seed}")
    print(f"Best final eval return: {best_score:.2f}")

    return best_seed


def main():
    episodes = 500
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

    print(f"Using custom Acrobot goal height: {GOAL_HEIGHT}")
    print("Standard Acrobot goal is around height >= 1.0")
    print("Maximum theoretical height is around 2.0")
    print("The red line in the generated videos shows the custom goal height.\n")

    for seed in seeds:
        for agent in agents:
            result = train_agent(
                agent_name=agent["agent_name"],
                backup_type=agent["backup_type"],
                seed=seed,
                env_name=ENV_NAME,
                goal_height=GOAL_HEIGHT,
                episodes=episodes,
                lr=5e-4,
                target_update_every=1000,
                epsilon_decay=0.997,
                min_epsilon=0.05,
                eval_every=10,
                eval_episodes=5,
                mellowmax_omega=5.0,
            )

            all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv("dqn_double_dqn_mellowmax_custom_goal_acrobot_results.csv", index=False)

    plot_results(results)
    print_summary(results)

    for agent in ["DQN", "Double DQN", "Mellowmax DQN"]:
        best_seed = select_best_model_seed(results, agent)

        model_path = f"{slugify(agent)}_goal{GOAL_HEIGHT}_seed{best_seed}_acrobot.pt"
        output_video = f"{slugify(agent)}_custom_goal_acrobot_demo.mp4"

        save_agent_video(
            model_path=model_path,
            output_path=output_video,
            env_name=ENV_NAME,
            goal_height=GOAL_HEIGHT,
            max_steps=500,
            fps=30,
            seed=999,
        )

    print("\nSaved files:")
    print("- dqn_double_dqn_mellowmax_custom_goal_acrobot_results.csv")
    print("- custom_goal_acrobot_summary.csv")
    print("- custom_goal_acrobot_train_return.png")
    print("- custom_goal_acrobot_eval_return.png")
    print("- custom_goal_acrobot_avg_max_q.png")
    print("- custom_goal_acrobot_eval_max_height.png")
    print("- dqn_custom_goal_acrobot_demo.mp4")
    print("- double_dqn_custom_goal_acrobot_demo.mp4")
    print("- mellowmax_dqn_custom_goal_acrobot_demo.mp4")


if __name__ == "__main__":
    main()