import os
from collections import defaultdict

# Ruta base donde están los directorios train, val y test
BASE_DIR = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\datasets\\5_dataset_reducido_etiquetas_bien"

# Carpetas a procesar
splits = ["train", "val", "test"]

# Clase 0: Gloves
# Clase 1: Helmet
# Clase 2: Human
# Clase 3: Safety Boot
# Clase 4: Safety Vest
# Clase 5: Glasses

# Diccionario para contar por split y clase
counts = {split: defaultdict(int) for split in splits}

for split in splits:
    labels_dir = os.path.join(BASE_DIR, split, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"Advertencia: no existe la carpeta {labels_dir}")
        continue
    
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(labels_dir, txt_file)
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id = line.strip().split()[0]  # Tomamos la primera columna
                    counts[split][class_id] += 1

# Mostramos resultados
for split in splits:
    print(f"\nSplit: {split}")
    for class_id, count in sorted(counts[split].items()):
        print(f"  Clase {class_id}: {count} EPIs")
