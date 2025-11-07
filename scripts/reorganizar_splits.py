import os
import shutil
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# === CONFIGURACIÓN ===
DUP_FILE = "dup_report.txt"
ORIG_BASE = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\imagenes_limpias"
NEW_BASE = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\imagenes_limpias_reducido"

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

print(f"Se han leído {len(groups)} grupos de duplicados")

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
        base, _ = os.path.splitext(os.path.basename(p))
        # Buscar el txt correspondiente en las tres posibles carpetas
        for split in ["train", "valid", "test"]:
            lbl_path = os.path.join(ORIG_BASE, split, "labels", f"{base}.txt")
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        count = sum(1 for _ in f)
                    if count > best_count:
                        best_count = count
                        best_path = p
                except Exception:
                    continue

    # Si ninguna tiene etiqueta, devolver la primera
    return best_path or img_paths[0]

# 4) Clasificar imágenes originales en grupos según split
splits = ["train", "valid", "test"]
split_counts = defaultdict(lambda: defaultdict(int))
split_to_paths = defaultdict(list)

for split in splits:
    split_dir = os.path.join(ORIG_BASE, split, "images")
    for dirpath, _, fnames in os.walk(split_dir):
        for fname in fnames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                fpath = os.path.join(dirpath, fname)
                gid = img_to_group.get(fpath)
                if gid is not None:
                    split_counts[gid][split] += 1
                else:
                    split_to_paths[split].append(fpath)

# 5) Seleccionar una imagen por grupo y asignar split
for gid, gpaths in enumerate(groups):
    split_freq = split_counts[gid]
    if not split_freq:
        continue
    majority_split = max(split_freq, key=split_freq.get)
    chosen = choose_best_image(gpaths)
    if chosen:
        split_to_paths[majority_split].append(chosen)

# 6) Copiar imágenes y etiquetas filtradas a la nueva estructura
for split, paths in split_to_paths.items():
    print(f"Copiando {len(paths)} imágenes a {split}/ ...")
    for p in tqdm(paths):
        relname = os.path.basename(p)
        base, _ = os.path.splitext(relname)

        # Copiar imagen
        out_img = os.path.join(NEW_BASE, split, "images", relname)
        shutil.copy2(p, out_img)

        # Buscar la etiqueta correspondiente en cualquier split
        label_found = False
        for s2 in ["train", "valid", "test"]:
            lbl_path = os.path.join(ORIG_BASE, s2, "labels", f"{base}.txt")
            if os.path.exists(lbl_path):
                out_lbl = os.path.join(NEW_BASE, split, "labels", f"{base}.txt")
                shutil.copy2(lbl_path, out_lbl)
                label_found = True
                break

        if not label_found:
            print(f"⚠️ No se encontró etiqueta para: {relname}")

# 7) Comprobación final
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(NEW_BASE, split, "images")
    lbl_dir = os.path.join(NEW_BASE, split, "labels")
    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
    missing = imgs - lbls
    print(f"\n📊 {split}: {len(imgs)} imágenes, {len(lbls)} etiquetas, {len(missing)} sin etiqueta")    

print("\n✅ Dataset limpio creado en:", NEW_BASE)
