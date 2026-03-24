import json
from pathlib import Path

root = Path("/home/kwonmc/jiyun/lerobot-VAI/dataset_git/libero_spatial_reproduce")
info_paths = sorted(root.glob("**/meta/info.json"))

total = 0
rows = []

for p in info_paths:
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(f"[SKIP] failed to read {p}: {e}")
        continue

    ep = data.get("total_episodes", None)
    if ep is None:
        print(f"[SKIP] no total_episodes in {p}")
        continue

    # 데이터셋 폴더(= .../<dataset_name>/meta/info.json 에서 <dataset_name>)
    dataset_dir = p.parent.parent  # meta/.. -> dataset folder
    rel = dataset_dir.relative_to(root)
    rows.append((str(rel), int(ep)))
    total += int(ep)

print(f"\nFound info.json: {len(info_paths)} files")
print("---- per-dataset total_episodes ----")
for name, ep in rows:
    print(f"{name}\t{ep}")
print("-----------------------------------")
print(f"SUM total_episodes = {total}")

