from ultralytics import YOLO

modelo = YOLO(
    "runs/detect/estacionamiento/IA-2doParcial-3/weights/best.pt"
)

print("Clases:", modelo.names)

resultados = modelo.predict(
    source="imagenes_ineditas",
    conf=0.25,
    imgsz=320,
    save=True
)

for resultado in resultados:
    print("\nImagen:", resultado.path)

    for caja in resultado.boxes:
        clase = int(caja.cls[0])
        confianza = float(caja.conf[0]) * 100

        print(
            modelo.names[clase],
            f"{confianza:.1f}%"
        )