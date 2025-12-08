"""
Train/test traditional schedulers and RL-based scheduler.
"""
import argparse
import copy
import os
import random
from typing import Dict, List, Optional
import numpy as np

from task import Task
from utils import (
    load_tasks_from_csv,
    calculate_metrics,
    print_metrics,
    compare_schedulers,
    clone_tasks,
    split_dataset,
)
from schedulers import FCFSScheduler, SJFScheduler, PriorityScheduler, RoundRobinScheduler
from scheduler_env import SchedulerEnv
from agent import DQNAgent
from rl_scheduler import RLScheduler
from ann_scheduler import ANNScheduler
from torch.utils.tensorboard import SummaryWriter


def train_rl_agent(
    tasks: List[Task],
    episodes: int = 100,
    max_queue_size: int = 10,
    state_size: int = 7,
    save_path: str = "models/rl_agent.pth",
    val_tasks: Optional[List[Task]] = None,
    eval_interval: int = 20,
    episode_task_count: int = 512,
    reward_scale: float = 50.0,
    completion_bonus: float = 1.0,
    idle_penalty: float = 0.05,
    learning_rate: float = 5e-4,
    gamma: float = 0.99,
    memory_size: int = 50000,
    batch_size: int = 128,
    target_update_freq: int = 1000,
    use_dueling: bool = True,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    tensorboard: bool = False,
    log_dir: str = "runs/rl",
) -> DQNAgent:
    print("\n" + "=" * 60)
    print("Training RL Agent")
    print("=" * 60)

    sample_count = min(len(tasks), episode_task_count)
    sample_env = SchedulerEnv(
        clone_tasks(tasks[:sample_count]),
        max_queue_size=max_queue_size,
        state_size=state_size,
        reward_scale=reward_scale,
        completion_bonus=completion_bonus,
        idle_penalty=idle_penalty,
        max_steps=sample_count * 2,
    )
    state_size_total = sample_env.state_dim
    action_size = sample_env.action_space_size

    # Agent hyperparameters are set by caller via CLI; use reasonable defaults here
    agent = DQNAgent(
        state_size=state_size_total,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay_episodes=max(20, episodes // 2),
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        use_dueling=use_dueling,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta=per_beta,
        per_eps=per_eps,
    )

    total_rewards = []
    episode_lengths = []
    loss_history = []
    best_eval_reward = float("-inf")
    best_state = None
    writer = SummaryWriter(log_dir) if tensorboard else None

    for episode in range(episodes):
        episode_size = min(len(tasks), episode_task_count)
        episode_tasks = random.sample(tasks, episode_size)
        env = SchedulerEnv(
            clone_tasks(episode_tasks),
            max_queue_size=max_queue_size,
            state_size=state_size,
            reward_scale=reward_scale,
            completion_bonus=completion_bonus,
            idle_penalty=idle_penalty,
            max_steps=episode_size * 2,
        )
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        episode_losses = []

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        if episode_losses:
            loss_history.append(np.mean(episode_losses))

        agent.decay_epsilon()

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(loss_history[-10:]) if loss_history else float("nan")
            print(
                f"Episode {episode + 1}/{episodes} - "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Avg Steps: {avg_length:.1f}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Avg Loss: {avg_loss:.4f}"
            )

        if val_tasks and (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_policy(
                agent,
                val_tasks,
                max_queue_size,
                state_size,
                reward_scale,
                completion_bonus,
                idle_penalty,
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_state = copy.deepcopy(agent.q_network.state_dict())
                print(f"  New best validation reward: {eval_reward:.2f}")

    if best_state is not None:
        agent.q_network.load_state_dict(best_state)
        agent.update_target_network()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final Avg Reward (last 10 eps): {np.mean(total_rewards[-10:]):.2f}")
    if best_state is not None:
        print(f"Best Validation Reward: {best_eval_reward:.2f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print("=" * 60 + "\n")
    return agent


def test_schedulers(
    tasks: List[Task],
    rl_agent: Optional[DQNAgent] = None,
    ann_scheduler: Optional[ANNScheduler] = None,
    max_queue_size: int = 10,
    state_size: int = 7,
    reward_scale: float = 50.0,
    completion_bonus: float = 1.0,
    idle_penalty: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    print("Testing FCFS Scheduler...")
    fcfs_tasks = FCFSScheduler().schedule(clone_tasks(tasks))
    results["FCFS"] = calculate_metrics(fcfs_tasks)
    print_metrics(results["FCFS"], "FCFS")

    print("Testing SJF Scheduler...")
    sjf_tasks = SJFScheduler().schedule(clone_tasks(tasks))
    results["SJF"] = calculate_metrics(sjf_tasks)
    print_metrics(results["SJF"], "SJF")

    print("Testing Priority Scheduler...")
    priority_tasks = PriorityScheduler().schedule(clone_tasks(tasks))
    results["Priority"] = calculate_metrics(priority_tasks)
    print_metrics(results["Priority"], "Priority")

    print("Testing Round Robin Scheduler...")
    rr_tasks = RoundRobinScheduler(time_quantum=1.0).schedule(clone_tasks(tasks))
    results["RoundRobin"] = calculate_metrics(rr_tasks)
    print_metrics(results["RoundRobin"], "RoundRobin")

    if rl_agent is not None:
        print("Testing RL Scheduler...")
        rl_tasks = RLScheduler(
            rl_agent,
            max_queue_size=max_queue_size,
            state_size=state_size,
            reward_scale=reward_scale,
            completion_bonus=completion_bonus,
            idle_penalty=idle_penalty,
        ).schedule(clone_tasks(tasks))
        results["RL (DQN)"] = calculate_metrics(rl_tasks)
        print_metrics(results["RL (DQN)"], "RL (DQN)")

    if ann_scheduler is not None:
        print("Testing ANN Scheduler...")
        ann_tasks = ann_scheduler.schedule(clone_tasks(tasks))
        results[f"ANN ({ann_scheduler.label_policy})"] = calculate_metrics(ann_tasks)
        print_metrics(results[f"ANN ({ann_scheduler.label_policy})"], f"ANN ({ann_scheduler.label_policy})")

    return results


def evaluate_policy(
    agent: DQNAgent,
    tasks: List[Task],
    max_queue_size: int,
    state_size: int,
    reward_scale: float,
    completion_bonus: float,
    idle_penalty: float,
) -> float:
    """Roll out agent on tasks and return cumulative reward."""
    env = SchedulerEnv(
        clone_tasks(tasks),
        max_queue_size=max_queue_size,
        state_size=state_size,
        reward_scale=reward_scale,
        completion_bonus=completion_bonus,
        idle_penalty=idle_penalty,
        max_steps=len(tasks) * 2,
    )
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated

    return total_reward


def print_rl_validation_metrics(
    agent: DQNAgent,
    tasks: List[Task],
    max_queue_size: int,
    state_size: int,
    reward_scale: float,
    completion_bonus: float,
    idle_penalty: float,
    label: str,
):
    """Evaluate RL scheduler on a dataset split and print metrics."""
    rl_scheduler = RLScheduler(
        agent,
        max_queue_size=max_queue_size,
        state_size=state_size,
        reward_scale=reward_scale,
        completion_bonus=completion_bonus,
        idle_penalty=idle_penalty,
    )
    completed = rl_scheduler.schedule(clone_tasks(tasks))
    metrics = calculate_metrics(completed)
    print_metrics(metrics, f"RL (DQN) [{label}]")


def main():
    parser = argparse.ArgumentParser(description="Task Scheduler with RL")
    parser.add_argument("--data", type=str, default="data/cloud_task_scheduling_dataset_20k.csv")
    parser.add_argument("--train", action="store_true", help="Force training of the RL agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--model", type=str, default="models/rl_agent.pth", help="Path to RL model")
    parser.add_argument("--load-model", action="store_true", help="Force loading an existing model")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL scheduler completely")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Testing split ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for dataset shuffling")
    parser.add_argument("--eval-interval", type=int, default=20, help="Validation eval interval (episodes)")
    parser.add_argument("--max-queue-size", type=int, default=10, help="Maximum queue size for RL")
    parser.add_argument("--state-size", type=int, default=7, help="Features per task in state representation")
    parser.add_argument("--reward-scale", type=float, default=50.0, help="Reward normalization factor")
    parser.add_argument("--completion-bonus", type=float, default=1.0, help="Reward bonus when a task completes")
    parser.add_argument("--idle-penalty", type=float, default=0.05, help="Penalty when scheduler is idle")
    parser.add_argument("--episode-task-count", type=int, default=512, help="Number of tasks sampled per training episode")
    parser.add_argument("--use-dueling", action="store_true", default=True, help="Use Dueling DQN architecture")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--memory-size", type=int, default=50000, help="Replay memory size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Target network update frequency (steps)")
    parser.add_argument("--use-per", action="store_true", default=True, help="Use Prioritized Experience Replay")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha parameter")
    parser.add_argument("--per-beta", type=float, default=0.4, help="PER beta parameter")
    parser.add_argument("--per-eps", type=float, default=1e-6, help="PER epsilon to avoid zero priority")
    parser.add_argument("--train-ann", action="store_true", help="Train ANN scheduler (imitation) on training set")
    parser.add_argument("--ann-epochs", type=int, default=10, help="Epochs for ANN training")
    parser.add_argument("--ann-policy", type=str, default="priority", choices=("priority", "sjf"), help="Label policy for ANN (priority or sjf)")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging during RL training")
    parser.add_argument("--log-dir", type=str, default="runs/rl", help="TensorBoard log directory")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    all_tasks = load_tasks_from_csv(args.data, arrival_time_mode="sequential")
    print(f"Loaded {len(all_tasks)} tasks")

    state_size = args.state_size

    train_tasks, val_tasks, test_tasks = split_dataset(
        all_tasks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    print(f"Training set: {len(train_tasks)} tasks")
    print(f"Validation set: {len(val_tasks)} tasks")
    print(f"Testing set: {len(test_tasks)} tasks")

    rl_agent: Optional[DQNAgent] = None
    if args.skip_rl:
        print("Skipping RL scheduler as requested.")
    else:
        model_exists = os.path.exists(args.model)
        if args.load_model and model_exists:
            print(f"\nLoading RL agent from {args.model}...")
            env = SchedulerEnv(
                clone_tasks(train_tasks[:100]),
                max_queue_size=args.max_queue_size,
                state_size=state_size,
                reward_scale=args.reward_scale,
                completion_bonus=args.completion_bonus,
                idle_penalty=args.idle_penalty,
            )
            state_dim = env.state_dim
            action_size = env.action_space_size
            rl_agent = DQNAgent(state_size=state_dim, action_size=action_size)
            rl_agent.load(args.model)
            print("Model loaded successfully!")
        elif args.train or not model_exists:
            if not model_exists and not args.train:
                print(f"\nModel {args.model} not found; training a new RL agent.")
            rl_agent = train_rl_agent(
                train_tasks,
                episodes=args.episodes,
                max_queue_size=args.max_queue_size,
                state_size=state_size,
                save_path=args.model,
                val_tasks=val_tasks,
                eval_interval=args.eval_interval,
                episode_task_count=args.episode_task_count,
                reward_scale=args.reward_scale,
                completion_bonus=args.completion_bonus,
                idle_penalty=args.idle_penalty,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                memory_size=args.memory_size,
                batch_size=args.batch_size,
                target_update_freq=args.target_update_freq,
                use_dueling=args.use_dueling,
                use_per=args.use_per,
                per_alpha=args.per_alpha,
                per_beta=args.per_beta,
                per_eps=args.per_eps,
                tensorboard=args.tensorboard,
                log_dir=args.log_dir,
            )
        elif model_exists:
            print(f"\nLoading existing RL agent from {args.model}...")
            env = SchedulerEnv(
                clone_tasks(train_tasks[:100]),
                max_queue_size=args.max_queue_size,
                state_size=state_size,
                reward_scale=args.reward_scale,
                completion_bonus=args.completion_bonus,
                idle_penalty=args.idle_penalty,
            )
            state_dim = env.state_dim
            action_size = env.action_space_size
            rl_agent = DQNAgent(state_size=state_dim, action_size=action_size)
            rl_agent.load(args.model)
            print("Model loaded successfully!")

    if rl_agent is not None and val_tasks:
        print("\nValidation evaluation for RL scheduler:")
        print_rl_validation_metrics(
            rl_agent,
            val_tasks,
            args.max_queue_size,
            state_size,
            args.reward_scale,
            args.completion_bonus,
            args.idle_penalty,
            "Validation",
        )

    # Optionally train ANN scheduler (imitation)
    ann_scheduler = None
    if args.train_ann:
        print("Training ANN scheduler (imitation)...")
        ann_scheduler = ANNScheduler(max_queue_size=args.max_queue_size, state_size=state_size, label_policy=args.ann_policy)
        ann_scheduler.fit(train_tasks, epochs=args.ann_epochs)
        print("ANN training complete.")

    print("\n" + "=" * 60)
    print("Testing All Schedulers")
    print("=" * 60)
    results = test_schedulers(
        test_tasks,
        rl_agent=rl_agent,
        ann_scheduler=ann_scheduler,
        max_queue_size=args.max_queue_size,
        state_size=state_size,
        reward_scale=args.reward_scale,
        completion_bonus=args.completion_bonus,
        idle_penalty=args.idle_penalty,
    )
    compare_schedulers(results)
    print("Done!")


if __name__ == "__main__":
    main()

