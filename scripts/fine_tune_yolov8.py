from ultralytics import YOLO
import torch

# Ruta al archivo de configuración de datos
data_path = "/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/data.yaml"

# Comprobar CPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detectada: {gpu_name}")
    device = 'cuda'
else:
    print("No se detectó GPU. Se entrenará en CPU")
    device = 'cpu'

# Cargar un modelo preentrenado (puedes cambiar 'yolov8n.pt' por otro: yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")

# Entrenar el modelo (fine-tuning)
model.train(
    data=data_path,        # Archivo YAML con rutas y clases
    epochs=2,              # Número de épocas de entrenamiento
    imgsz=640,             # Tamaño de las imágenes (640 recomendado)
    batch=16,              # Tamaño del batch (ajusta según la RAM/VRAM)
    device='cuda',         # Si tienes GPU, usa 'cuda'; si no, usa 'cpu'
    name="yolov8n_finetuned",  # Nombre del experimento
    project="/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/runs"  # Carpeta donde guardar resultados
)

# Evaluar el modelo en el conjunto de validación
metrics = model.val(data=data_path)
print(metrics)

# Probar el modelo entrenado en el conjunto de test
results = model.predict(source="/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/test/images", save=True, conf=0.5)
print("Predicciones completadas. Archivos guardados en:", model.predictor.save_dir)