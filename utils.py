"""
Utility functions for loading data, evaluation, and visualization.
"""
import csv
import os
from typing import List, Dict, Tuple
import numpy as np
from task import Task


def load_tasks_from_csv(filepath: str, arrival_time_mode: str = 'sequential') -> List[Task]:
    """
    Load tasks from CSV file.
    
    Args:
        filepath: Path to CSV file
        arrival_time_mode: 'sequential' (tasks arrive one after another) or 
                          'random' (random arrival times)
    
    Returns:
        List of Task objects
    """
    tasks = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        current_time = 0.0
        
        for row in reader:
            task = Task(
                task_id=int(row['Task_ID']),
                cpu_usage=float(row['CPU_Usage (%)']),
                ram_usage=float(row['RAM_Usage (MB)']),
                disk_io=float(row['Disk_IO (MB/s)']),
                network_io=float(row['Network_IO (MB/s)']),
                priority=int(row['Priority']),
                vm_id=int(row['VM_ID']),
                execution_time=float(row['Execution_Time (s)']),
                arrival_time=current_time
            )
            tasks.append(task)
            
            if arrival_time_mode == 'sequential':
                # Tasks arrive one after another with small gap
                current_time += 0.1
            elif arrival_time_mode == 'random':
                # Random arrival times
                current_time += np.random.exponential(0.5)
    
    return tasks


def calculate_metrics(tasks: List[Task]) -> Dict[str, float]:
    """
    Calculate scheduling performance metrics.
    
    Args:
        tasks: List of completed tasks
    
    Returns:
        Dictionary with metrics
    """
    if not tasks:
        return {}
    
    completed_tasks = [t for t in tasks if t.is_completed()]
    if not completed_tasks:
        return {}
    
    waiting_times = [t.get_waiting_time() for t in completed_tasks]
    turnaround_times = [t.get_turnaround_time() for t in completed_tasks]
    response_times = [t.get_response_time() for t in completed_tasks]
    
    metrics = {
        'avg_waiting_time': np.mean(waiting_times),
        'avg_turnaround_time': np.mean(turnaround_times),
        'avg_response_time': np.mean(response_times),
        'max_waiting_time': np.max(waiting_times),
        'max_turnaround_time': np.max(turnaround_times),
        'throughput': len(completed_tasks) / max([t.completion_time for t in completed_tasks]) if completed_tasks else 0,
        'cpu_utilization': sum([t.execution_time for t in completed_tasks]) / max([t.completion_time for t in completed_tasks]) if completed_tasks else 0,
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], scheduler_name: str):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"Scheduler: {scheduler_name}")
    print(f"{'='*60}")
    print(f"Average Waiting Time:     {metrics.get('avg_waiting_time', 0):.2f}s")
    print(f"Average Turnaround Time:   {metrics.get('avg_turnaround_time', 0):.2f}s")
    print(f"Average Response Time:     {metrics.get('avg_response_time', 0):.2f}s")
    print(f"Max Waiting Time:          {metrics.get('max_waiting_time', 0):.2f}s")
    print(f"Max Turnaround Time:       {metrics.get('max_turnaround_time', 0):.2f}s")
    print(f"Throughput:                {metrics.get('throughput', 0):.4f} tasks/sec")
    print(f"CPU Utilization:           {metrics.get('cpu_utilization', 0):.2%}")
    print(f"{'='*60}\n")


def compare_schedulers(results: Dict[str, Dict[str, float]]):
    """Compare results from multiple schedulers."""
    # Nicely formatted fixed-width table
    schedulers = list(results.keys())
    metrics = ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'throughput']

    # sanitize and determine column widths
    schedames_clean = [s.replace('\n', ' ') for s in schedulers]
    col_width = 18
    metric_col = 26
    total_width = metric_col + len(schedulers) * col_width

    print("\n" + "=" * total_width)
    title = "SCHEDULER COMPARISON"
    print(title.center(total_width))
    print("=" * total_width)

    # Header
    header = f"{'Metric':<{metric_col}}"
    for s in schedames_clean:
        short = (s[: col_width - 2] + '..') if len(s) > col_width - 2 else s
        header += f"{short:^{col_width}}"
    print(header)
    print("-" * total_width)

    # Rows
    for metric in metrics:
        row = f"{metric:<{metric_col}}"
        for sched in schedulers:
            value = results[sched].get(metric, 0)
            if 'time' in metric:
                cell = f"{value:>8.2f}s"
            else:
                cell = f"{value:>8.4f}"
            row += f"{cell:^{col_width}}"
        print(row)

    print("=" * total_width + "\n")


def clone_tasks(tasks: List[Task]) -> List[Task]:
    """Create a fresh copy of each task."""
    cloned = []
    for task in tasks:
        cloned.append(
            Task(
                task_id=task.task_id,
                cpu_usage=task.cpu_usage,
                ram_usage=task.ram_usage,
                disk_io=task.disk_io,
                network_io=task.network_io,
                priority=task.priority,
                vm_id=task.vm_id,
                execution_time=task.execution_time,
                arrival_time=task.arrival_time,
            )
        )
    return cloned


def split_dataset(
    tasks: List[Task],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """
    Split tasks into train/val/test subsets using all data.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    total = len(tasks)
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    def build_subset(idxs):
        return clone_tasks([tasks[i] for i in idxs])

    train_subset = build_subset(train_indices)
    val_subset = build_subset(val_indices)
    test_subset = build_subset(test_indices)

    return train_subset, val_subset, test_subset

