import time
import os
import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("model/resnet18_partial_best.pth", map_location=device)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

folder = "dataset/val/horse_hitam"
image_name = os.listdir(folder)[0]
image_path = os.path.join(folder, image_name)

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    model(image)

times = []

for _ in range(100):
    start = time.perf_counter()

    with torch.no_grad():
        model(image)

    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_latency = sum(times) / len(times)
avg_fps = 1000 / avg_latency

print(f"Average latency: {avg_latency:.2f} ms")
print(f"Average FPS: {avg_fps:.2f}")
