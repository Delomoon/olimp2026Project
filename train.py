
from pathlib import Path
import random
import shutil

from ultralytics import YOLO

# Путь к исходному датасету
SOURCE = Path("CNRPark")

# Папка для подготовленной выборки
DEST = Path("CNRPark_yolo")

random.seed(42)

# 80% train, 10% val, 10% test
splits = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

classes = ["busy", "free"]
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

# Автоматически собираем изображения из A и B
for split in splits:
    for cls in classes:
        (DEST / split / cls).mkdir(
            parents=True, exist_ok=True
        )

for cls in classes:
    images = []

    for camera in ["A", "B"]:
        folder = SOURCE / camera / cls

        if not folder.exists():
            raise FileNotFoundError(
                f"Не найдена папка: {folder}"
            )

        images.extend(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in extensions
        )

    random.shuffle(images)

    n_train = int(len(images) * 0.8)
    n_val = int(len(images) * 0.1)

    groups = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, files in groups.items():
        target = DEST / split / cls

        for i, src in enumerate(files):
            # Уникальные имена, чтобы не было конфликтов
            dst = target / f"{cls}_{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        print(f"{split}/{cls}: {len(files)} изображений")

# Обучаем YOLO11n для классификации
model = YOLO("yolo11n-cls.pt")

model.train(
    data=str(DEST.resolve()),
    epochs=50,
    imgsz=224,
    batch=32,
    patience=10,
    seed=42,
)

# Оцениваем качество на тестовой выборке
model.val(
    data=str(DEST.resolve()),
    split="test",
    imgsz=224,
    batch=32,
)
