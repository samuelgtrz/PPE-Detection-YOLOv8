import cv2
import os
import torch
import numpy as np
import shutil
from tqdm import tqdm

# ========== CONFIGURACIÓN ==========
DATASET_PATH = r"/mnt/netapp2/Store_uni/home/usc/cursos/curso1589/construccion_limpiada"
OUTPUT_PATH  = r"/mnt/netapp2/Store_uni/home/usc/cursos/curso1589/limpia_ultima_version"

THRESHOLD = 5.0  # umbral de diferencia media entre imágenes 0–255
IMG_SIZE = (256, 256)

IMAGES_PATH = os.path.join(DATASET_PATH, "images")
LABELS_PATH = os.path.join(DATASET_PATH, "labels")

OUT_IMG = os.path.join(OUTPUT_PATH, "images")
OUT_LBL = os.path.join(OUTPUT_PATH, "labels")
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ========== CARGAR TODAS LAS IMÁGENES EN MEMORIA ==========
images = [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(valid_ext)]
print(f"📸 {len(images)} imágenes encontradas")

img_tensors = []

for img_name in tqdm(images, desc="Cargando imágenes"):
    img = cv2.imread(os.path.join(IMAGES_PATH, img_name), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = torch.tensor(img, dtype=torch.float32)
    img_tensors.append(img.flatten())   # vector 256×256 → 65536 valores

# Tensor final en GPU
imgs_gpu = torch.stack(img_tensors).cuda()    # (N, 65536)

# ========== CALCULAR DIFERENCIA ENTRE TODAS ==========
print("⚡ Calculando diferencias en GPU...")

# Distancia media absoluta entre imágenes
# torch.cdist → L2 por defecto → cambiamos a L1 manualmente
diff_matrix = torch.cdist(imgs_gpu, imgs_gpu, p=1) / (IMG_SIZE[0] * IMG_SIZE[1])

# Evitamos autocomparación
diff_matrix.fill_diagonal_(9999)

# ========== DETECTAR DUPLICADOS ==========
duplicates = set()

for i, img_name in enumerate(images):
    if img_name in duplicates:
        continue

    row = diff_matrix[i].cpu().numpy()
    dup_ids = np.where(row < THRESHOLD)[0]

    for j in dup_ids:
        duplicates.add(images[j])

print(f"🗑️ Duplicados encontrados: {len(duplicates)}")

# ========== COPIAR IMÁGENES FINALES ==========
saved = 0
for img in images:
    if img in duplicates:
        continue

    label = os.path.splitext(img)[0] + ".txt"

    shutil.copy(os.path.join(IMAGES_PATH, img),
                os.path.join(OUT_IMG, img))

    if os.path.exists(os.path.join(LABELS_PATH, label)):
        shutil.copy(os.path.join(LABELS_PATH, label),
                    os.path.join(OUT_LBL, label))
    saved += 1

print(f"✅ Dataset final: {saved} imágenes únicas")
print(f"📁 Guardado en {OUTPUT_PATH}")
