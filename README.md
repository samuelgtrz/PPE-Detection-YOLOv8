
# 🦺 Detección de Equipos de Protección Personal (PPE) con YOLOv8

Este proyecto implementa un sistema de **detección automática de equipos de protección personal (EPP)** utilizando **YOLOv8**, con el objetivo de identificar si las personas en entornos laborales cumplen con las normas de seguridad (casco, guantes, chaleco, botas, gafas, etc.).

---

## 📸 Ejemplos de inferencia

A continuación se muestran algunos resultados obtenidos tras el fine-tuning del modelo:

| Ejemplo 1 | Ejemplo 2 |
|------------|------------|
| ![Inferencia 1](./resultados_ejecuciones/4_ppe_yolov8_finetuned_etiquetas_mod/predicciones/Aitin0586_jpg.rf.8cad5fb756af5f7f315c920e6115014e.jpg) | ![Inferencia 2](.resultados_ejecuciones/4_ppe_yolov8_finetuned_etiquetas_mod/predicciones/image_160_jpg.rf.2ec56a1123be881b57114fd04821d6cb.jpg) |

> Las imágenes de inferencia se incluyen únicamente con fines demostrativos.

---

## 🧠 Modelo entrenado

El modelo fue entrenado usando **YOLOv8n (versión nano)** de Ultralytics como punto de partida, realizando un **fine-tuning** sobre un dataset propio reducido y depurado.

- **Modelo base:** `yolov8n.pt`
- **Framework:** PyTorch / Ultralytics YOLOv8
- **Formato exportado:** `.pt` (PyTorch) y `.onnx` (ONNX)
- **Número de clases finales:** 6  
