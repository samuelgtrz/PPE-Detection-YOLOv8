import os

# Ruta base del dataset
for i in ["test", "train", "valid"]:
    print(f"--- Procesando conjunto: {i} ---")
    base_path = f"C:/Users/sam20/OneDrive/Documentos/IA/CuartoIA/Proyecto_Integrador_2/Proyecto/Construction PPE.v3i.yolov8/{i}"

    # Subcarpetas de imágenes y etiquetas
    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")

    # Extensiones válidas de imágenes
    valid_exts = [".jpg", ".jpeg", ".png"]

    # Contar imágenes
    images = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].lower() in valid_exts]
    num_images = len(images)

    # Contar etiquetas
    labels = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
    num_labels = len(labels)

    # Comparar imágenes y etiquetas
    print(f"Imágenes encontradas: {num_images}")
    print(f"Archivos de etiquetas: {num_labels}")


    # Ver si hay imágenes sin etiqueta o viceversa
    images_sin_label = []
    labels_sin_imagen = []

    for img in images:
        base_name = os.path.splitext(img)[0]
        label_file = base_name + ".txt"
        if label_file not in labels:
            images_sin_label.append(img)

    for lbl in labels:
        base_name = os.path.splitext(lbl)[0]
        image_found = any(base_name == os.path.splitext(img)[0] for img in images)
        if not image_found:
            labels_sin_imagen.append(lbl)

    if not images_sin_label and not labels_sin_imagen:
        print(f"Todas las imágenes tienen su etiqueta correspondiente en {i}.")
    else:
        if images_sin_label:
            print(f"{len(images_sin_label)} imágenes no tienen archivo .txt:")
            for f in images_sin_label[:5]:
                print("   -", f)
        if labels_sin_imagen:
            print(f"{len(labels_sin_imagen)} etiquetas no tienen imagen:")
            for f in labels_sin_imagen[:5]:
                print("   -", f)
print("FIN")

