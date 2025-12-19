
# 🦺 Detección de Equipos de Protección Personal (PPE) con YOLOv8

Este proyecto implementa un sistema de **detección automática de equipos de protección personal (EPP)** utilizando **YOLOv8**, con el objetivo de identificar si las personas en entornos laborales cumplen con las normas de seguridad (casco, guantes, chaleco, botas, gafas, humano).

---

## 📸 Ejemplos de inferencia

Algunos resultados obtenidos tras el fine-tuning del modelo:

| Ejemplo 1 |
|------------|
| ![Inferencia 2](./fine_tunings/7_comparacion_runs_v8m/detect_20vs30_patience/predict_20/ppe_0961_jpg.rf.7494b8f700fd5908178be6e66587956d.jpg)

| Ejemplo 2 |
|------------|
| ![Inferencia 1](./fine_tunings/7_comparacion_runs_v8m/detect_20vs30_patience/predict_20/00000075_jpg.rf.713a362429dec71962d36a7dc0aac654.jpg)


---

## 🧠 Modelo entrenado

El modelo fue entrenado usando **YOLOv8** de Ultralytics como punto de partida, realizando un **fine-tuning** con este **[dataset](https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo)** ya etiquetado de **Roboflow** , posteriormente reducido y depurado por nosotros.

- **Modelo base:** `yolov8n.pt`
- **Framework:** PyTorch / Ultralytics YOLOv8
- **Formato exportado:** `.pt` (PyTorch) y `.onnx` (ONNX)
- **Número de clases detectadas:** 6  
