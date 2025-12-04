from ultralytics import YOLO
from inference import get_model as get_rf_model
import os
import csv

# =========================================================
# CONFIGURACIÓN DE MODELOS (AQUÍ SOLO AÑADES MÁS SI QUIERES)
# =========================================================

MODELS = [
    {
        "nombre": "yolo_finetuned",
        "tipo": "local",
        "path": r"C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\resultados_ejecuciones\\5_comparacion_runs_modelos\\detect_m_no_freeze\\ppe_yolov8_finetuned\\weights\\best.pt",
    },
    {
        "nombre": "roboflow_masks",
        "tipo": "roboflow",
        "model_id": "masks-xsrnl/2"   # <<< este es tu modelo de Roboflow
    }
]

IMAGE_PATH = r"C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\datasets\\5_dataset_reducido_etiquetas_bien\\test\\images\\image_160_jpg.rf.2ec56a1123be881b57114fd04821d6cb.jpg"

OUTPUT_CSV = r"C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\resultados_ejecuciones\\inferencias_individuales\\predicciones_inferencia.csv"


# =========================================================
# CARGA DE MODELOS (solo una vez)
# =========================================================

def cargar_modelos():
    modelos_cargados = []

    for cfg in MODELS:
        if cfg["tipo"] == "local":
            modelo = YOLO(cfg["path"])
        else:  # roboflow
            modelo = get_rf_model(cfg["model_id"])

        modelos_cargados.append({
            "nombre": cfg["nombre"],
            "tipo": cfg["tipo"],
            "modelo": modelo
        })

        print(f"✔ Modelo cargado: {cfg['nombre']}")

    return modelos_cargados


# =========================================================
# FUNCIÓN DE INFERENCIA UNIFICADA
# =========================================================

def inferir_con_modelos(modelos, imagen, writer):
    for m in modelos:
        nombre = m["nombre"]
        modelo = m["modelo"]

        # ========== YOLO LOCAL ==========
        if m["tipo"] == "local":
            results = modelo.predict(imagen, verbose=False, conf=0.7)

            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls)
                    cls_name = r.names[cls_id]
                    conf = float(b.conf)

                    writer.writerow([os.path.basename(imagen),
                                     nombre,
                                     cls_id,
                                     cls_name,
                                     conf])

        # ========== ROBOFLOW ==========
        else:
            results = modelo.infer(imagen)[0]
            for det in results.predictions:
                writer.writerow([
                    os.path.basename(imagen),
                    nombre,
                    det.class_id,
                    det.class_name,
                    det.confidence
                ])


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

if __name__ == "__main__":

    modelos = cargar_modelos()

    # Abrimos CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imagen", "modelo", "class_id", "class_name", "confidence"])

        inferir_con_modelos(modelos, IMAGE_PATH, writer)

    print(f"\n✔ Archivo generado: {OUTPUT_CSV}")
