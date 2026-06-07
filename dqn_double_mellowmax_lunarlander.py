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


ENV_NAME = "LunarLander-v3"


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


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
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_env(env_name=ENV_NAME, render_mode=None):
    """
    Uses LunarLander-v3 by default.
    If your installed Gymnasium version does not have v3, change ENV_NAME to LunarLander-v2.
    """
    if render_mode is None:
        return gym.make(env_name)
    return gym.make(env_name, render_mode=render_mode)


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
    """
    mellowmax(x) = 1 / omega * log(mean(exp(omega * x)))

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


def evaluate_policy(
    policy_net,
    env_name,
    seed,
    episodes,
    device,
):
    env = make_env(env_name)

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
                    device=device
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
    episodes: int = 600,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 5e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 2_000,
    train_every: int = 1,
    target_update_every: int = 1000,
    min_epsilon: float = 0.05,
    epsilon_decay: float = 0.995,
    eval_every: int = 10,
    eval_episodes: int = 5,
    mellowmax_omega: float = 5.0,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {agent_name} on {device}")

    env = make_env(env_name)
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
            )
        else:
            eval_return = np.nan
            eval_len = np.nan

        eval_returns.append(eval_return)
        eval_lengths.append(eval_len)

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

            print(
                f"{agent_name} | seed={seed} | episode={episode + 1:04d} | "
                f"train_return_25={recent_train:7.1f} | "
                f"eval_return={recent_eval:7.1f} | "
                f"epsilon={epsilon:.3f} | "
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
    })

    model_filename = f"{slugify(agent_name)}_seed{seed}_lunarlander.pt"
    torch.save(policy_net.state_dict(), model_filename)
    print(f"Saved model: {model_filename}")

    return result


def save_agent_video(
    model_path,
    output_path,
    env_name=ENV_NAME,
    max_steps=1000,
    fps=30,
    seed=999,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_name, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    frames = []

    state, _ = env.reset(seed=seed)
    done = False
    total_return = 0.0
    step = 0

    while not done and step < max_steps:
        frame = env.render()
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
        step += 1

    for _ in range(30):
        frame = env.render()
        frames.append(frame)

    env.close()

    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Saved video: {output_path}")
    print(f"Video return: {total_return:.2f}")
    print(f"Video length: {step} steps")


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

    plt.title("LunarLander-v3")
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
        filename="lunarlander_train_return.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="eval_return",
        ylabel="Evaluation return, smoothed over 10 eval points",
        filename="lunarlander_eval_return.png",
        only_eval_points=True,
    )

    plot_metric(
        results,
        metric="avg_max_q",
        ylabel="Average max Q(s, a), smoothed over 10 episodes",
        filename="lunarlander_avg_max_q.png",
        only_eval_points=False,
    )

    plot_metric(
        results,
        metric="episode_length",
        ylabel="Episode length, smoothed over 10 episodes",
        filename="lunarlander_episode_length.png",
        only_eval_points=False,
    )


def print_summary(results: pd.DataFrame):
    eval_results = results.dropna(subset=["eval_return"])

    summary_train = (
        results
        .groupby("agent")
        .agg(
            final_50_train_return=("train_return", lambda x: np.mean(x.tail(50))),
            final_50_avg_max_q=("avg_max_q", lambda x: np.mean(x.tail(50))),
            final_50_episode_length=("episode_length", lambda x: np.mean(x.tail(50))),
            best_train_return=("train_return", "max"),
        )
        .reset_index()
    )

    summary_eval = (
        eval_results
        .groupby("agent")
        .agg(
            final_10_eval_return=("eval_return", lambda x: np.mean(x.tail(10))),
            best_eval_return=("eval_return", "max"),
            final_10_eval_length=("eval_length", lambda x: np.mean(x.tail(10))),
        )
        .reset_index()
    )

    summary = summary_train.merge(summary_eval, on="agent", how="left")

    print("\nSummary:")
    print(summary.to_string(index=False))

    summary.to_csv("lunarlander_summary.csv", index=False)


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
    episodes = 600

    # Na szybki eksperyment zostaw jeden seed.
    # Jeśli masz więcej czasu, ustaw np. seeds = [0, 1, 2].
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
    print("LunarLander has continuous state observations and discrete actions.")
    print("Return above 200 is usually considered solved in this environment.")
    print("Training DQN, Double DQN and Mellowmax DQN.\n")

    for seed in seeds:
        for agent in agents:
            result = train_agent(
                agent_name=agent["agent_name"],
                backup_type=agent["backup_type"],
                seed=seed,
                env_name=ENV_NAME,
                episodes=episodes,
                batch_size=128,
                gamma=0.99,
                lr=5e-4,
                buffer_size=100_000,
                learning_starts=2_000,
                train_every=1,
                target_update_every=1000,
                epsilon_decay=0.995,
                min_epsilon=0.05,
                eval_every=10,
                eval_episodes=5,
                mellowmax_omega=5.0,
            )

            all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv(
        "dqn_double_dqn_mellowmax_lunarlander_results.csv",
        index=False
    )

    plot_results(results)
    print_summary(results)

    for agent in ["DQN", "Double DQN", "Mellowmax DQN"]:
        best_seed = select_best_model_seed(results, agent)

        model_path = f"{slugify(agent)}_seed{best_seed}_lunarlander.pt"
        output_video = f"{slugify(agent)}_lunarlander_demo.mp4"

        save_agent_video(
            model_path=model_path,
            output_path=output_video,
            env_name=ENV_NAME,
            max_steps=1000,
            fps=30,
            seed=999,
        )

    print("\nSaved files:")
    print("- dqn_double_dqn_mellowmax_lunarlander_results.csv")
    print("- lunarlander_summary.csv")
    print("- lunarlander_train_return.png")
    print("- lunarlander_eval_return.png")
    print("- lunarlander_avg_max_q.png")
    print("- lunarlander_episode_length.png")
    print("- dqn_lunarlander_demo.mp4")
    print("- double_dqn_lunarlander_demo.mp4")
    print("- mellowmax_dqn_lunarlander_demo.mp4")


if __name__ == "__main__":
    main()