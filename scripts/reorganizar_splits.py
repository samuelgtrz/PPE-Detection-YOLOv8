import os
import shutil
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import random

# === CONFIGURACIÓN ===
DUP_FILE = r"C:\Users\Usuario\OneDrive - Universidade de Santiago de Compostela\GRIA\4º CURSO. 1º CUADRIMESTRE\Proxecto Integrador II\Proyecto-Integrador-de-IA\datasets\6_dataset_reducido_etiquetas_bien\dup_report.txt"
ORIG_BASE = r"C:\Users\Usuario\OneDrive - Universidade de Santiago de Compostela\GRIA\4º CURSO. 1º CUADRIMESTRE\Proxecto Integrador II\Proyecto-Integrador-de-IA\datasets\6_dataset_reducido_etiquetas_bien"
IMG_DIR = os.path.join(ORIG_BASE, "images")
LABEL_DIR = os.path.join(ORIG_BASE, "labels")
NEW_BASE = r"C:\Users\Usuario\OneDrive - Universidade de Santiago de Compostela\GRIA\4º CURSO. 1º CUADRIMESTRE\Proxecto Integrador II\Proyecto-Integrador-de-IA\datasets\7_dataset_reorganizado_sin_duplicados"

# Ratios para train/val/test
RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}

# ======================
# Crear estructura de carpetas destino
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(NEW_BASE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(NEW_BASE, split, "labels"), exist_ok=True)

# 1) Leer grupos de duplicados del informe
groups = []
current_group = []
with open(DUP_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "":
            if current_group:
                groups.append(current_group)
                current_group = []
        elif line.startswith("GROUP"):
            continue
        else:
            current_group.append(line)
if current_group:
    groups.append(current_group)

print(f"📑 Se han leído {len(groups)} grupos de duplicados")

# 2) Mapear cada imagen a su grupo
img_to_group = {}
for i, g in enumerate(groups):
    for p in g:
        img_to_group[p] = i

# 3) Heurística: quedarse con la imagen que tenga más EPIs (líneas en el txt)
def choose_best_image(img_paths):
    if len(img_paths) == 1:
        return img_paths[0]

    best_path = None
    best_count = -1

    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        lbl_path = os.path.join(LABEL_DIR, f"{base}.txt")
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                if count > best_count:
                    best_count = count
                    best_path = p
            except Exception:
                continue

    return best_path or img_paths[0]

# 4) Crear lista única sin duplicados
all_images = [os.path.join(IMG_DIR, f)
              for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]

unique_images = []
seen_groups = set()

for img_path in all_images:
    gid = img_to_group.get(img_path)
    if gid is None:
        # no pertenece a ningún grupo duplicado
        unique_images.append(img_path)
    elif gid not in seen_groups:
        chosen = choose_best_image(groups[gid])
        unique_images.append(chosen)
        seen_groups.add(gid)

print(f"✅ Total de imágenes únicas tras eliminar duplicados: {len(unique_images)}")

# 5) Dividir en train/val/test
random.shuffle(unique_images)
n = len(unique_images)
n_train = int(RATIOS["train"] * n)
n_valid = int(RATIOS["valid"] * n)

splits = {
    "train": unique_images[:n_train],
    "valid": unique_images[n_train:n_train+n_valid],
    "test": unique_images[n_train+n_valid:]
}

# 6) Copiar imágenes y etiquetas a la nueva estructura
for split, paths in splits.items():
    print(f"\n📦 Copiando {len(paths)} imágenes a {split}/ ...")
    for p in tqdm(paths):
        relname = os.path.basename(p)
        base, _ = os.path.splitext(relname)

        # Copiar imagen
        out_img = os.path.join(NEW_BASE, split, "images", relname)
        shutil.copy2(p, out_img)

        # Copiar etiqueta correspondiente
        lbl_path = os.path.join(LABEL_DIR, f"{base}.txt")
        if os.path.exists(lbl_path):
            out_lbl = os.path.join(NEW_BASE, split, "labels", f"{base}.txt")
            shutil.copy2(lbl_path, out_lbl)
        else:
            print(f"⚠️ No se encontró etiqueta para: {relname}")

# 7) Comprobación final
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(NEW_BASE, split, "images")
    lbl_dir = os.path.join(NEW_BASE, split, "labels")
    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
    missing = imgs - lbls
    print(f"\n📊 {split}: {len(imgs)} imágenes, {len(lbls)} etiquetas, {len(missing)} sin etiqueta")

print("\n✅ Dataset limpio y reorganizado creado en:", NEW_BASE)