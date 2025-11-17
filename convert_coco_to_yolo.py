import os

LABEL_DIR = "dataset_yolo/labels"

def check_yolo_labels(label_dir):
    errors = 0
    for split in ["train", "val"]:
        split_dir = os.path.join(label_dir, split)
        for file in os.listdir(split_dir):
            if not file.endswith(".txt"):
                continue
            path = os.path.join(split_dir, file)
            with open(path, "r") as f:
                lines = f.readlines()
            if not lines:
                print(f"⚠️ Empty label file: {path}")
                errors += 1
                continue
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"❌ Wrong number of elements in {path}, line {i+1}: {line.strip()}")
                    errors += 1
                    continue
                cls, x, y, w, h = parts
                try:
                    cls = int(cls)
                    x, y, w, h = map(float, (x, y, w, h))
                except ValueError:
                    print(f"❌ Non-numeric value in {path}, line {i+1}: {line.strip()}")
                    errors += 1
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"❌ Value out of range in {path}, line {i+1}: {line.strip()}")
                    errors += 1

    if errors == 0:
        print("✅ All YOLO label files are correctly formatted!")
    else:
        print(f"⚠️ Found {errors} issues in label files.")

check_yolo_labels(LABEL_DIR)
