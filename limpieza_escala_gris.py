import cv2
import numpy as np
import os
from itertools import combinations
from tqdm import tqdm
import shutil

# === CONFIGURACIÓN ===
DATASET_PATH = "C:/Users/Roque/Documents/USC/Curso_4/Cuatri_1/Proyecto_Integrador/Work/Proyecto-Integrador-de-IA/imagenes_limpias"   # carpeta original con 'images' y 'labels'
OUTPUT_PATH = "C:/Users/Roque/Documents/USC/Curso_4/Cuatri_1/Proyecto_Integrador"      # carpeta destino
THRESHOLD = 5.0                     # menor -> más estricto
RESIZE_TO = (64, 64)                # tamaño para comparación

# === RUTAS ===
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
LABELS_PATH = os.path.join(DATASET_PATH, "labels")

OUTPUT_IMAGES = os.path.join(OUTPUT_PATH, "images")
OUTPUT_LABELS = os.path.join(OUTPUT_PATH, "labels")

# Crear carpetas destino si no existen
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# === FUNCIÓN PARA CALCULAR DIFERENCIA ENTRE IMÁGENES ===
def image_diff(img1_path, img2_path, resize_to=RESIZE_TO):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return np.inf
    img1 = cv2.resize(img1, resize_to)
    img2 = cv2.resize(img2, resize_to)
    return np.mean(np.abs(img1 - img2))

# === LISTAR IMÁGENES ===
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
images = [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(valid_ext)]

print(f"Se encontraron {len(images)} imágenes en '{IMAGES_PATH}'")

# === DETECTAR DUPLICADOS ===
duplicates = set()
for img1, img2 in tqdm(combinations(images, 2), total=len(images)*(len(images)-1)//2):
    if img1 in duplicates or img2 in duplicates:
        continue
    diff = image_diff(os.path.join(IMAGES_PATH, img1), os.path.join(IMAGES_PATH, img2))
    if diff < THRESHOLD:
        duplicates.add(img2)

# === COPIAR IMÁGENES Y LABELS ===
copied = 0
for img_name in images:
    if img_name in duplicates:
        continue

    # Emparejar label con el mismo nombre, solo cambiando extensión
    label_name = os.path.splitext(img_name)[0] + ".txt"

    src_img = os.path.join(IMAGES_PATH, img_name)
    src_label = os.path.join(LABELS_PATH, label_name)
    dst_img = os.path.join(OUTPUT_IMAGES, img_name)
    dst_label = os.path.join(OUTPUT_LABELS, label_name)

    shutil.copy(src_img, dst_img)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)

    copied += 1

print(f"\n✅ Dataset limpio creado con {copied} imágenes únicas.")
print(f"🗂️ Guardado en: {OUTPUT_PATH}")
