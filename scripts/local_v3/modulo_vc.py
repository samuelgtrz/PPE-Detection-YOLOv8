from ultralytics import YOLO
import os
import csv



# CARGAR MODELO
def model_loader(MODE_PATH):
    model = YOLO(MODE_PATH)
    print("Modelo cargado correctamente.")
    return model

def execute_model(model, IMAGE_PATH, OUTPUT_CSV):
    # ABRIR CSV PARA GUARDAR RESULTADOS

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imagen", "clase_id", "clase_nombre", "confianza"])

        # Si IMAGE_PATH es un directorio, itera archivos; si es un archivo, procesa solo ese.
        if os.path.isdir(IMAGE_PATH):
            image_files = [os.path.join(IMAGE_PATH, n) for n in os.listdir(IMAGE_PATH)]
        else:
            image_files = [IMAGE_PATH]

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            results = model.predict(img_path, verbose=False, conf=0.7)

            for r in results:
                boxes = r.boxes
                for b in boxes:
                    cls_id = int(b.cls)
                    cls_name = r.names[cls_id]
                    conf = float(b.conf)
                    writer.writerow([img_name, cls_id, cls_name, conf])

    print(f"Archivo generado: {OUTPUT_CSV}")

