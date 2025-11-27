# Task Scheduler with Reinforcement Learning

Implements classic CPU scheduling algorithms alongside a reinforcement-learning based scheduler trained with a Deep Q-Network (DQN).

## Features
- FCFS, SJF, Priority, and Round Robin schedulers
- RL scheduler that learns from the provided dataset
- Metrics: waiting time, turnaround time, response time, throughput, CPU utilization

## Setup
```bash
pip install -r requirements.txt
```

## Usage
- Test traditional schedulers:
  ```bash
  python main.py --data data/cloud_task_scheduling_dataset_20k.csv --test-size 1000
  ```
- Train the RL agent:
  ```bash
  python main.py --train --episodes 100
  ```
- Load a trained model and include RL in testing:
  ```bash
  python main.py --load-model --model models/rl_agent.pth --test-size 1000
  ```

## Data Requirements
CSV columns: `Task_ID, CPU_Usage (%), RAM_Usage (MB), Disk_IO (MB/s), Network_IO (MB/s), Priority, VM_ID, Execution_Time (s)`.

## Project Structure
- `task.py` – Task model
- `schedulers.py` – Traditional schedulers
- `scheduler_env.py` – Environment for RL training
- `agent.py` – DQN agent
- `rl_scheduler.py` – Wrapper for trained agent
- `utils.py` – Data loading & metrics
- `main.py` – CLI for training/testing

