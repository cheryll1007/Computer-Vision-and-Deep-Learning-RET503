import os
import time
import copy
import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

import matplotlib.pyplot as plt


# =========================
# KONFIGURASI
# =========================

DATA_DIR = "dataset"
RESULT_DIR = "results"
MODEL_DIR = "model"

EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# SEED
# =========================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# =========================
# TRANSFORM
# =========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Jumlah data train:", len(train_dataset))
print("Jumlah data validation:", len(val_dataset))


# =========================
# MODEL
# =========================

def create_model(mode):

    if mode == "feature":
        print("\nLoading ResNet-18 pretrained ImageNet...")

        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze semua layer
        for param in model.parameters():
            param.requires_grad = False

        # Ganti FC
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        optimizer = optim.Adam(
            model.fc.parameters(),
            lr=1e-3
        )

    elif mode == "partial":
        print("\nLoading ResNet-18 pretrained ImageNet...")

        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze semua layer terlebih dahulu
        for param in model.parameters():
            param.requires_grad = False

        # Buka layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Ganti FC
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        optimizer = optim.Adam([
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3
            }
        ])

    elif mode == "scratch":
        print("\nLoading ResNet-18 from scratch...")

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3
        )

    else:
        raise ValueError("Mode tidak dikenal.")

    model = model.to(device)

    return model, optimizer


# =========================
# TRAINING
# =========================

def train_model(mode):

    print("\n" + "=" * 60)
    print("TRAINING MODE:", mode.upper())
    print("=" * 60)

    model, optimizer = create_model(mode)

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    history = []

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    start_total = time.time()

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # -------------------------
        # TRAIN
        # -------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time
        })

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # Simpan model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

    total_time = time.time() - start_total

    # Load best model
    model.load_state_dict(best_model_weights)

    # Simpan model
    model_path = os.path.join(
        MODEL_DIR,
        f"resnet18_{mode}_best.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    # Cari epoch pertama dengan val acc >= 90%
    epoch_90 = None

    for row in history:

        if row["val_acc"] >= 0.90:
            epoch_90 = row["epoch"]
            break

    # Simpan history CSV
    csv_path = os.path.join(
        RESULT_DIR,
        f"history_{mode}.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "epoch_time"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    # Grafik accuracy
    epochs = [x["epoch"] for x in history]
    train_accs = [x["train_acc"] for x in history]
    val_accs = [x["val_acc"] for x in history]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accs,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accs,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"ResNet-18 - {mode.upper()}")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"accuracy_{mode}.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "mode": mode,
        "best_val_acc": best_val_acc,
        "epoch_90": epoch_90,
        "training_time": total_time
    }


# =========================
# JALANKAN 3 MODE
# =========================

results = []

for mode in [
    "feature",
    "partial",
    "scratch"
]:

    result = train_model(mode)

    results.append(result)


# =========================
# SIMPAN HASIL 3 MODE
# =========================

summary_path = os.path.join(
    RESULT_DIR,
    "results_summary.csv"
)

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "best_val_acc",
            "epoch_90",
            "training_time"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

for result in results:

    print(
        f"{result['mode'].upper()} | "
        f"Best Val Acc: {result['best_val_acc']:.4f} | "
        f"Epoch >= 90%: {result['epoch_90']} | "
        f"Training Time: {result['training_time']:.2f} s"
    )

print("\nHasil tersimpan di folder:")
print("results/")
print("model/")
