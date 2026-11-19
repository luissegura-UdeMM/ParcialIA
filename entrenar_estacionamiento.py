from ultralytics import YOLO

if __name__ == "__main__":

    modelo = YOLO("yolov8n.pt")

    modelo.train(
        data="dataset-estacionamiento/data.yaml",
        epochs=1,
        imgsz=320,
        batch=16,
        workers=0,
        project="estacionamiento",
        name="IA-2doParcial"
    )