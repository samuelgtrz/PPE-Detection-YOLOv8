import os
import random
import shutil

# Este script lo usamos porque al crear el dataset nuevo con las etiquetass actualizadas a 6 clases, tenemos un dataset compacto sin dividir en train test y val. 
# Este script lo que hace es dividir aleatoriamente ese dataset en esas tres partes y copiar las imágenes y etiquetas correspondientes a cada carpeta.

DATASET_DIR = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\6_provisional"   # carpeta original con /images y /labels
OUTPUT_DIR = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\6_cuatro_clases"  # carpeta donde se guardará el dataset dividido

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6 #lo que hace es verificar que la suma de TRAIN_RATIO, VAL_RATIO y TEST_RATIO sea igual a 1.0 dentro de una pequeña tolerancia (1e-6).



# ================================
# CREAR ESTRUCTURA DE CARPETAS
# ================================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

# ================================
# LISTAR Y EMPAREJAR IMÁGENES
# ================================
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]

print(f"Total de imágenes encontradas: {len(images)}")

# Mezclar aleatoriamente
random.shuffle(images)

# ================================
# DIVIDIR EN TRAIN / VAL / TEST
# ================================
train_cut = int(len(images) * TRAIN_RATIO)
val_cut = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

train_imgs = images[:train_cut]
val_imgs = images[train_cut:val_cut]
test_imgs = images[val_cut:]

def mover(lista_imgs, split_name):
    for img_name in lista_imgs:
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_img = os.path.join(IMG_DIR, img_name)
        src_lbl = os.path.join(LBL_DIR, label_name)

        dst_img = os.path.join(OUTPUT_DIR, split_name, "images", img_name)
        dst_lbl = os.path.join(OUTPUT_DIR, split_name, "labels", label_name)

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            print(f"Etiqueta no encontrada para {img_name}, solo imagen copiada.")

# ================================
# MOVER ARCHIVOS
# ================================
mover(train_imgs, "train")
mover(val_imgs, "val")
mover(test_imgs, "test")

print("División completada.")
print(f"Train: {len(train_imgs)} imágenes")
print(f"Val:   {len(val_imgs)} imágenes")
print(f"Test:  {len(test_imgs)} imágenes")
print(f"Dataset guardado en: {OUTPUT_DIR}")
