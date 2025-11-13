from ultralytics import YOLO
import os
import csv

# =============================
# CONFIGURACIÓN
# =============================
MODE_PATH = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\resultados_ejecuciones\\5_comparacion_runs_modelos\\detect_m_no_freeze\\ppe_yolov8_finetuned\\weights\\best.pt"
IMAGE_PATH = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\5_dataset_reducido_etiquetas_bien\\test\\images\\image_160_jpg.rf.2ec56a1123be881b57114fd04821d6cb.jpg"
OUTPUT_CSV = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\resultados_ejecuciones\\inferencias_individuales\\predicciones_inferencia.csv"

# =============================
# CARGAR MODELO
# =============================
model = YOLO(MODE_PATH)
print("Modelo cargado correctamente.")

# =============================
# ABRIR CSV PARA GUARDAR RESULTADOS
# =============================
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

print(f"✔ Archivo generado: {OUTPUT_CSV}")
