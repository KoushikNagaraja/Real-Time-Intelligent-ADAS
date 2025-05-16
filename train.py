from ultralytics import YOLO

model_path="C:\\Users\\Koushik N\\Desktop\\Yolo11_Final_Project\\models\\yolo11s.pt"

model=YOLO(model_path)

traffic_data= "C:\\Users\\Koushik N\\Desktop\\Yolo11_Final_Project\\traffic_data.yaml"

EPOCHS=50
IMAGE_SIZE=320
DEVICE_TYPE='cpu'

output=model.train(data=traffic_data,
                   epochs=EPOCHS,
                   imgsz=IMAGE_SIZE,
                   device=DEVICE_TYPE)

path=model.export(format="onnx")