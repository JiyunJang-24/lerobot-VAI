import json
import os
import csv
import argparse

def analyze_success_rate(json_path, csv_path=None):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"\nAnalyzing: {json_path}")
    print(f"{'Task ID':<10} | {'Task Group':<20} | {'Success Rate (%)':<20} | {'Successes/Total':<15}")
    print("-" * 75)

    per_task = data.get('per_task', [])

    total_successes = 0
    total_episodes = 0
    csv_rows = []

    for task in per_task:
        task_id = task.get('task_id')
        task_group = task.get('task_group')
        successes = task.get('metrics', {}).get('successes', [])

        n_success = sum(1 for s in successes if s is True)
        n_total = len(successes)

        success_rate = (n_success / n_total) * 100 if n_total > 0 else 0

        print(f"{task_id:<10} | {task_group:<20} | {success_rate:<20.2f} | {n_success}/{n_total}")

        csv_rows.append([task_id, task_group, f"{success_rate:.2f}", f"{n_success}/{n_total}"])

        total_successes += n_success
        total_episodes += n_total

    print("-" * 75)
    overall_rate = (total_successes / total_episodes) * 100 if total_episodes > 0 else 0
    print(f"{'OVERALL':<33} | {overall_rate:<20.2f} | {total_successes}/{total_episodes}")

    if csv_path:
        header = ['Task ID', 'Task Group', 'Success Rate (%)', 'Successes/Total']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_rows)
            writer.writerow(['OVERALL', 'All', f"{overall_rate:.2f}", f"{total_successes}/{total_episodes}"])
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze evaluation results.')
    parser.add_argument('json_path', type=str, help='Path to eval_info.json')
    parser.add_argument('--csv', type=str, help='Output CSV path')

    args = parser.parse_args()
    analyze_success_rate(args.json_path, args.csv)
