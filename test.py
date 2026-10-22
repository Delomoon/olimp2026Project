from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

results = model.predict(
    source="test_image.jpg",
    imgsz=224
)

for result in results:
    print(result.names[result.probs.top1])
    print(result.probs.top1conf)