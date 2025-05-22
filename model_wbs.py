import os
from datasets import Dataset, concatenate_datasets
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from sklearn.metrics import classification_report
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 내 이미지 불러오기
my_image_folder = "./my_images"
my_data = []
for filename in os.listdir(my_image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(my_image_folder, filename)).convert("RGB")
        my_data.append({
            "image": transform(img),
            "label": 1
        })
my_dataset = Dataset.from_list(my_data)

# 지브리 이미지 불러오기
ghibli_image_folder = "./ghibli_images"
ghibli_data = []
for filename in os.listdir(ghibli_image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(ghibli_image_folder, filename)).convert("RGB")
        ghibli_data.append({
            "image": transform(img),
            "label": 0
        })
ghibli_dataset = Dataset.from_list(ghibli_data)

# 데이터셋 결합 및 셔플
combined = concatenate_datasets([ghibli_dataset, my_dataset]).shuffle(seed=42)

# PyTorch Dataset 변환
class TorchImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if isinstance(image, list):
            image = torch.tensor(image)
        return image, item["label"]

# 학습/검증 데이터 분할
train_size = int(0.8 * len(combined))
train_dataset = TorchImageDataset(combined.select(range(train_size)))
val_dataset = TorchImageDataset(combined.select(range(train_size, len(combined))))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 모델 정의
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

model_path = "efficientnet_mine_vs_ghibli.pt"

# 모델 학습 or 불러오기
if os.path.exists(model_path):
    print("💾 저장된 모델 불러오는 중...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print("🧠 모델 학습 중...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), model_path)
    print("✅ 학습 완료 및 모델 저장됨")

# 검증 데이터 평가
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

print("\n=== 검증 데이터 평가 ===")
print(classification_report(y_true, y_pred, target_names=["Ghibli", "Mine"]))

# 테스트 이미지 예측
test_image_folder = "./test_images"
test_data = []
filenames = []

for filename in os.listdir(test_image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(test_image_folder, filename)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        test_data.append(img_tensor)
        filenames.append(filename)

print("\n=== test_images 예측 결과 ===")
for i, img_tensor in enumerate(test_data):
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        label_str = "Mine" if pred == 1 else "Ghibli"
        print(f"{filenames[i]} → 예측: {label_str}")
