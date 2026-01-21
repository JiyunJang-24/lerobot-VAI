import json
import os

def analyze_success_rate(json_path):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"{'Task ID':<10} | {'Task Group':<20} | {'Success Rate (%)':<20} | {'Successes/Total':<15}")
    print("-" * 75)

    per_task = data.get('per_task', [])

    total_successes = 0
    total_episodes = 0

    for task in per_task:
        task_id = task.get('task_id')
        task_group = task.get('task_group')
        successes = task.get('metrics', {}).get('successes', [])

        n_success = sum(1 for s in successes if s is True)
        n_total = len(successes)

        success_rate = (n_success / n_total) * 100 if n_total > 0 else 0

        print(f"{task_id:<10} | {task_group:<20} | {success_rate:<20.2f} | {n_success}/{n_total}")

        total_successes += n_success
        total_episodes += n_total

    print("-" * 75)
    overall_rate = (total_successes / total_episodes) * 100 if total_episodes > 0 else 0
    print(f"{'OVERALL':<33} | {overall_rate:<20.2f} | {total_successes}/{total_episodes}")

if __name__ == "__main__":
    json_file = "/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/eval/2026-01-20/17-01-52_smolvla_spatial_vanilla_reproduce/eval_info.json"
    analyze_success_rate(json_file)
