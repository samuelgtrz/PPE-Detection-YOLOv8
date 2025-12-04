import os
import shutil



# Carpeta donde están tus imágenes y labels
DATASET_DIR = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\imagenes_reducidas"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

# Carpeta destino del dataset limpio
OUT_DIR = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\6_provisional"
OUT_IMG = os.path.join(OUT_DIR, "images")
OUT_LBL = os.path.join(OUT_DIR, "labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# ================================================================
# MAPEOS Y REGLAS
# ================================================================

# Clases finales (nuevo orden)
FINAL_CLASSES = [
    "Gloves",        # 0
    "Helmet",        # 1
    "Safety Boot",   # 2
    "Safety Vest",   # 3
]

# Mapeo desde clases antiguas -> nuevas clases
CLASS_MAP = {
    "Gloves": "Gloves",

    "Helmet": "Helmet",

    "Safety Boot": "Safety Boot",

    "Safety Vest": "Safety Vest",

    # Clases prohibidas (se eliminan)
    "Glasses": None,
    "Human": None,
}

# ================================================================
# ETIQUETAS ORIGINALES DEL YAML (asegúrate de mantener este orden)
# ================================================================

ORIGINAL_CLASSES = [
    'Gloves', 'Helmet', 'Human', 'Safety Boot', 'Safety Vest',
    'glasses'
]

ORIG_INDEX_TO_NAME = {i: n for i, n in enumerate(ORIGINAL_CLASSES)}
NEW_NAME_TO_INDEX = {name: i for i, name in enumerate(FINAL_CLASSES)}

# ================================================================
# PROCESAR
# ================================================================

print("=== Iniciando limpieza de dataset ===")

kept = 0
removed_images = 0

for label_file in os.listdir(LBL_DIR):

    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(LBL_DIR, label_file)
    img_name = os.path.splitext(label_file)[0] + ".jpg"
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"⚠️ Imagen no encontrada para {label_file}, saltando.")
        continue

    new_lines = []

    # Leer anotaciones
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            coords = parts[1:]

            old_class_name = ORIG_INDEX_TO_NAME[cls_id]
            new_class_name = CLASS_MAP.get(old_class_name, None)

            # Si la clase está marcada como prohibida (None)
            if new_class_name is None:
                continue

            # Clase válida
            new_class_id = NEW_NAME_TO_INDEX[new_class_name]
            new_lines.append(str(new_class_id) + " " + " ".join(coords))

    # Si después de limpiar no queda ninguna anotación válida, borrar imagen y txt
    if len(new_lines) == 0:
        removed_images += 1
        os.remove(label_path)
        os.remove(img_path)
        continue

    # Guardar en dataset limpio
    shutil.copy(img_path, os.path.join(OUT_IMG, img_name))

    with open(os.path.join(OUT_LBL, label_file), "w") as f:
        f.write("\n".join(new_lines))

    kept += 1

print("Limpieza completada.")
print(f"Archivos conservados: {kept}")
print(f"Imágenes eliminadas por contener solo clases que se tienen que borrar: {removed_images}")

print("\nNuevo YAML sugerido:")
print("--------------------------------------")
print("nc: 6")
print("names:", FINAL_CLASSES)
print("--------------------------------------")
