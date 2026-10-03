import os
import random
import shutil

random.seed(42)

classes = ["horse_hitam", "chariot_hitam"]

for cls in classes:
    source = f"dataset_raw/{cls}"
    train = f"dataset/train/{cls}"
    val = f"dataset/val/{cls}"

    images = [
        f for f in os.listdir(source)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    train_images = images[:80]
    val_images = images[80:]

    for img in train_images:
        shutil.copy2(
            os.path.join(source, img),
            os.path.join(train, img)
        )

    for img in val_images:
        shutil.copy2(
            os.path.join(source, img),
            os.path.join(val, img)
        )

    print(f"{cls}: {len(train_images)} train, {len(val_images)} validation")
