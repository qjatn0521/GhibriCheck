import os
from PIL import Image
from datasets import load_dataset

# Hugging Face Token
my_token = "MYKEY"

# 1. 500개만 불러오기 + 훈련셋만 사용
ghibli = load_dataset("Nechintosh/ghibli", token=my_token, split="train[:500]")
ghibli = ghibli.train_test_split(test_size=0.2, seed=42)["train"]

# 2. 저장할 폴더 만들기
save_dir = "ghibli_images"
os.makedirs(save_dir, exist_ok=True)

# 3. 이미지 저장
for idx, example in enumerate(ghibli):
    img = example["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")

    save_path = os.path.join(save_dir, f"ghibli_{idx:04}.jpg")
    img.save(save_path)
    print(f"Saved: {save_path}")
