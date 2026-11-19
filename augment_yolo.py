#!/usr/bin/env python3
"""
Data augmentation para datasets YOLO (imágenes .jpg + etiquetas .txt).

Lee todas las imágenes de una carpeta, genera N variantes aumentadas por
imagen aplicando transformaciones aleatorias y transforma los bounding
boxes de las etiquetas para que sigan siendo válidos.

Formato de etiqueta de entrada/salida (YOLO):
    clase  cx  cy  w  h        (todo normalizado 0..1)

Uso:
    python augment_yolo.py --input ./imagenes --output ./aumentadas --num 5
    python augment_yolo.py -i ./imagenes -o ./aumentadas -n 10 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------- etiquetas
def cargar_etiquetas(path: Path, w: int, h: int):
    """Lee un .txt YOLO y devuelve (clase, corners_px) por objeto.

    corners_px: 4 esquinas [(x, y)] del bounding box en píxeles.
    """
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, x2 = (cx - bw / 2) * w, (cx + bw / 2) * w
        y1, y2 = (cy - bh / 2) * h, (cy + bh / 2) * h
        labels.append((cls, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
    return labels


def guardar_etiquetas(path: Path, labels, w: int, h: int):
    """Reconvierte las esquinas (px) a formato YOLO normalizado y las guarda."""
    lines = []
    for cls, corners in labels:
        xs = [min(max(x, 0), w) for x, _ in corners]
        ys = [min(max(y, 0), h) for _, y in corners]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 1e-4 or bh <= 1e-4:      # objeto que quedó fuera de la imagen
            continue
        cx = (((x1 + x2) / 2) % w) / w
        cy = (((y1 + y2) / 2) % h) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def corners_a_yolo(labels, M, w: int, h: int):
    """Aplica una matriz afín 2x3 a las esquinas de todos los boxes."""
    nuevas = []
    for cls, corners in labels:
        pts = np.array(corners, dtype=np.float64).reshape(-1, 1, 2)
        pts = cv2.transform(pts, M).reshape(-1, 2)
        nuevas.append((cls, [tuple(p) for p in pts]))
    return nuevas


# ----------------------------------------------------------------------------- transformaciones
def t_flip_h(img, labels, rng, w, h):
    img = cv2.flip(img, 1)
    M = np.float32([[-1, 0, w], [0, 1, 0]])
    return img, corners_a_yolo(labels, M, w, h)


def t_flip_v(img, labels, rng, w, h):
    img = cv2.flip(img, 0)
    M = np.float32([[1, 0, 0], [0, -1, h]])
    return img, corners_a_yolo(labels, M, w, h)


def t_rotate(img, labels, rng, w, h):
    ang = rng.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), \
        corners_a_yolo(labels, M, w, h)


def t_translate(img, labels, rng, w, h):
    tx = rng.uniform(-0.12, 0.12) * w
    ty = rng.uniform(-0.12, 0.12) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), \
        corners_a_yolo(labels, M, w, h)


def t_scale(img, labels, rng, w, h):
    s = rng.uniform(1.1, 1.4)         # zoom (recorta bordes, boxes se agrandan)
    M = np.float32([[s, 0, 0.5 * w * (1 - s)], [0, s, 0.5 * h * (1 - s)]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), \
        corners_a_yolo(labels, M, w, h)


def t_hsv(img, labels, rng, w, h):
    """Jitter de color: no afecta a los boxes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = np.clip(hsv[..., 0] + rng.integers(-15, 16), 0, 179)  # tono
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)  # saturación
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3), 0, 255)  # brillo
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), labels


def t_brightness_contrast(img, labels, rng, w, h):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    return np.clip(img, 0, 255).astype(np.uint8), labels


def t_blur(img, labels, rng, w, h):
    k = rng.integers(3, 8) * 2 + 1    # impar: 3..15
    return cv2.GaussianBlur(img, (k, k), 0), labels


def t_noise(img, labels, rng, w, h):
    ruido = rng.normal(0, rng.uniform(6, 15), img.shape)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8), labels


def t_grayscale(img, labels, rng, w, h):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR), labels  # 3 canales (YOLO lo exige)


# probabilidad de aplicar cada transformación en cada variante generada
AUMENTACIONES = {
    t_flip_h: 0.5,
    t_flip_v: 0.1,
    t_rotate: 0.4,
    t_translate: 0.4,
    t_scale: 0.3,
    t_hsv: 0.7,
    t_brightness_contrast: 0.7,
    t_blur: 0.3,
    t_noise: 0.3,
    t_grayscale: 0.1,
}


def generar_variante(img, labels, rng, w, h):
    """Aplica un subconjunto aleatorio de aumentaciones (≥ 1 geométrica opcional)."""
    aplica = [(f, p) for f, p in AUMENTACIONES.items() if rng.random() < p]
    if not aplica:
        aplica = [(rng.choice(list(AUMENTACIONES)), 1.0)]
    out_img, out_lbl = img, labels
    for f, _ in aplica:
        out_img, out_lbl = f(out_img, out_lbl, rng, w, h)
    return out_img, out_lbl


# ----------------------------------------------------------------------------- programa
def main():
    ap = argparse.ArgumentParser(
        description="Data augmentation para datasets YOLO (jpg + txt).")
    ap.add_argument("-i", "--input", required=True, help="Carpeta con imágenes y .txt")
    ap.add_argument("-o", "--output", required=True, help="Carpeta de salida")
    ap.add_argument("-n", "--num", type=int, default=5,
                    help="Variantes aumentadas por imagen (default: 5)")
    ap.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    ap.add_argument("--copiar-originales", action="store_true",
                    help="También copiar las imágenes originales a la salida")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    entrada = Path(args.input)
    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)

    imagenes = sorted(p for p in entrada.iterdir()
                      if p.suffix.lower() in IMG_EXTS and not p.name.startswith("classes"))
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en {entrada}")

    total = 0
    for img_path in imagenes:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[aviso] No se pudo leer {img_path.name}, se salta")
            continue
        h, w = img.shape[:2]
        stem = img_path.stem
        txt_path = img_path.with_suffix(".txt")
        labels = cargar_etiquetas(txt_path, w, h)

        for i in range(args.num):
            out_img, out_lbl = generar_variante(img.copy(), labels, rng, w, h)
            nombre = f"{stem}_aug{i + 1}"
            cv2.imwrite(str(salida / f"{nombre}{img_path.suffix}"), out_img)
            guardar_etiquetas(salida / f"{nombre}.txt", out_lbl, w, h)
            total += 1

        if args.copiar_originales:
            # copia directa del archivo (sin recodificar), imagen + etiqueta
            shutil.copy2(img_path, salida / img_path.name)
            if txt_path.exists():
                shutil.copy2(txt_path, salida / txt_path.name)

        print(f"{img_path.name}: {args.num} variantes "
              f"({len(labels)} objetos)")

    print(f"\n{total} imágenes aumentadas guardadas en {salida}")


if __name__ == "__main__":
    main()
