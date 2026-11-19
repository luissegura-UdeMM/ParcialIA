#!/usr/bin/env python3
"""
Reparte un dataset YOLO (imágenes + etiquetas .txt) en train/val/test.

Dada una carpeta con imágenes (.jpg, .png, ...) y sus etiquetas .txt,
reparte aleatoriamente las imágenes siguiendo la estructura estándar de
YOLOv5/YOLOv8:
    <salida>/images/train   (70% por defecto)
    <salida>/images/val     (15%)
    <salida>/images/test    (el resto)
y las etiquetas .txt en las carpetas paralelas:
    <salida>/labels/train  <salida>/labels/val  <salida>/labels/test

Uso:
    python repartir_dataset.py --input ./dataset --output ./dataset
    python repartir_dataset.py -i ./dataset -o ./dataset --seed 42 --porcentajes 70 15 15
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(
        description="Reparte imágenes YOLO y sus txt en train/val/test.")
    ap.add_argument("-i", "--input", required=True,
                    help="Carpeta con las imágenes y sus .txt")
    ap.add_argument("-o", "--output", required=True,
                    help="Carpeta donde crear images/train, images/val e images/test")
    ap.add_argument("-p", "--porcentajes", type=int, nargs=3,
                    default=[70, 15, 15], metavar=("TRAIN", "VAL", "TEST"),
                    help="Porcentajes para train/val/test (default: 70 15 15)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla aleatoria para un reparto reproducible")
    args = ap.parse_args()

    train_pct, val_pct, _test_pct = args.porcentajes
    if train_pct + val_pct > 100:
        raise SystemExit("La suma de train y val no puede superar el 100 %")

    origen = Path(args.input)
    if not origen.is_dir():
        raise SystemExit(f"No existe la carpeta de entrada: {origen}")

    # solo imágenes; cada una se empareja con su .txt homónimo
    imagenes = sorted(p for p in origen.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS
                      and not p.name.startswith("classes"))
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en {origen}")

    rng = random.Random(args.seed)
    rng.shuffle(imagenes)

    salida = Path(args.output)
    conjuntos = ("train", "val", "test")
    destino_imagenes = {c: salida / "images" / c for c in conjuntos}
    destino_etiquetas = {c: salida / "labels" / c for c in conjuntos}
    for rutas in (destino_imagenes, destino_etiquetas):
        for ruta in rutas.values():
            ruta.mkdir(parents=True, exist_ok=True)

    n = len(imagenes)
    n_train = n * train_pct // 100
    n_val = n * val_pct // 100
    asignados = {
        "train": imagenes[:n_train],
        "val": imagenes[n_train:n_train + n_val],
        "test": imagenes[n_train + n_val:],
    }

    accion = shutil.copy2
    for conjunto, lista in asignados.items():
        for img_path in lista:
            accion(str(img_path),
                   destino_imagenes[conjunto] / img_path.name)
            txt = img_path.with_suffix(".txt")
            hay_etiqueta = txt.exists()
            if hay_etiqueta:                       # la imagen puede no tener etiqueta
                accion(str(txt), destino_etiquetas[conjunto] / txt.name)
            print(f"{conjunto:5s} img <- {img_path.name}")
            if hay_etiqueta:
                print(f"      lbl <- {txt.name}")

    resumen = (f"Total: {n} imágenes | "
               f"train={len(asignados['train'])} "
               f"({len(asignados['train']) / n:.0%}), "
               f"val={len(asignados['val'])} ({len(asignados['val']) / n:.0%}), "
               f"test={len(asignados['test'])} ({len(asignados['test']) / n:.0%})")
    print(f"\n{resumen}")
    print(f"Reparto guardado en {salida / 'images'} y {salida / 'labels'}")


if __name__ == "__main__":
    main()
