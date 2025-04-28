import os
import numpy as np
from pathlib import Path

base_path = Path("/home/temp_id/shared_data/capture")
target_value = 32767

# (name, index) pair 전부 수집 후 날짜 기준으로 정렬
all_pairs = []
for name_dir in base_path.iterdir():
    if not name_dir.is_dir():
        continue
    for index_dir in name_dir.iterdir():
        if not index_dir.is_dir():
            continue
        all_pairs.append((name_dir.name, index_dir.name, index_dir.stat().st_mtime))

# 날짜순 정렬
all_pairs.sort(key=lambda x: x[2])

for name, index, _ in all_pairs:
    data_path = base_path / name / index / "contact/data.npy"
    if not data_path.exists():
        continue
    
    data = np.load(data_path)

    is_bad = False
    for i in range(15):
        if np.all(data[:, i] == target_value):
            is_bad = True
    
    if not is_bad:
        import pdb; pdb.set_trace()
        print(f"Good: {name} {index}")