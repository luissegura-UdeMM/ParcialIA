from ultralytics import YOLO

if __name__ == "__main__":
    modelo = YOLO(
        "runs/detect/estacionamiento/IA-2doParcial-3/weights/best.pt"
    )

    modelo.train(
        data="data_perros.yaml",
        epochs=5,
        imgsz=320,
        batch=16,
        workers=0,
        project="perros",
        name="IA-2doParcial"
    )