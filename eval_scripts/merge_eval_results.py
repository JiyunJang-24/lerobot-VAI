import argparse
import csv
import json
import os

def get_task_data(json_path):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return None

    with open(json_path) as f:
        data = json.load(f)

    task_results = {}
    per_task = data.get("per_task", [])

    for task in per_task:
        task_id = task.get("task_id")
        task_group = task.get("task_group")
        successes = task.get("metrics", {}).get("successes", [])

        n_success = sum(1 for s in successes if s is True)
        n_total = len(successes)
        success_rate = (n_success / n_total) * 100 if n_total > 0 else 0

        task_results[task_id] = {
            "task_group": task_group,
            "success_rate": success_rate,
            "n_success": n_success,
            "n_total": n_total,
        }

    return task_results

def merge_results(vanilla_json, basis_json, output_csv):
    vanilla_data = get_task_data(vanilla_json)
    basis_data = get_task_data(basis_json)

    if vanilla_data is None or basis_data is None:
        return

    task_ids = sorted(set(vanilla_data.keys()) | set(basis_data.keys()))

    # New header format to match LIBERO-long.csv requirements
    header = ["model", "smolVLA", "smolVLA + basis(ours)"]
    rows = []

    print(f"{'Task':<25} | {'smolVLA (%)':<15} | {'smolVLA+Basis (%)':<20} | {'Diff (%)':<10}")
    print("-" * 75)

    total_v_success = 0
    total_v_episodes = 0
    total_b_success = 0
    total_b_episodes = 0

    for tid in task_ids:
        v = vanilla_data.get(tid, {"success_rate": 0, "n_success": 0, "n_total": 0})
        b = basis_data.get(tid, {"success_rate": 0, "n_success": 0, "n_total": 0})

        diff = b["success_rate"] - v["success_rate"]

        # Row name format: "libero long task <ID>"
        task_label = f"libero long task {tid}"

        print(f"{task_label:<25} | {v['success_rate']:<15.2f} | {b['success_rate']:<20.2f} | {diff:+.2f}")

        rows.append(
            [
                task_label,
                f"{v['success_rate']:.1f}",
                f"{b['success_rate']:.1f}",
            ]
        )

        total_v_success += v["n_success"]
        total_v_episodes += v["n_total"]
        total_b_success += b["n_success"]
        total_b_episodes += b["n_total"]

    v_overall = (total_v_success / total_v_episodes * 100) if total_v_episodes > 0 else 0
    b_overall = (total_b_success / total_b_episodes * 100) if total_b_episodes > 0 else 0

    print("-" * 75)
    print(f"{'average':<25} | {v_overall:<15.2f} | {b_overall:<20.2f} | {b_overall - v_overall:+.2f}")

    # Add average row to CSV
    rows.append(
        [
            "average",
            f"{v_overall:.1f}",
            f"{b_overall:.1f}",
        ]
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nMerged results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Vanilla and Basis evaluation results for plotting.")
    parser.add_argument("--vanilla", type=str, required=True, help="Path to vanilla eval_info.json")
    parser.add_argument("--basis", type=str, required=True, help="Path to basis eval_info.json")
    parser.add_argument("--output", type=str, default="merged_results.csv", help="Output CSV path")

    args = parser.parse_args()
    merge_results(args.vanilla, args.basis, args.output)
