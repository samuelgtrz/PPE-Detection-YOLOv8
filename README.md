
# 🦺 Detección de Equipos de Protección Personal (PPE) con YOLOv8

Este proyecto implementa un sistema de **detección automática de equipos de protección personal (EPP)** utilizando **YOLOv8**, con el objetivo de identificar si las personas en entornos laborales cumplen con las normas de seguridad (casco, guantes, chaleco, botas, gafas, humano).

---

## 📸 Ejemplos de inferencia

Algunos resultados obtenidos tras el fine-tuning del modelo:

| Ejemplo 1 |
|------------|
| ![Inferencia 1](./resultados_ejecuciones/4_ppe_yolov8_finetuned_etiquetas_mod/predicciones/image_160_jpg.rf.2ec56a1123be881b57114fd04821d6cb.jpg) 

| Ejemplo 2 |
|------------|
| ![Inferencia 2](./resultados_ejecuciones/4_ppe_yolov8_finetuned_etiquetas_mod/predicciones/Aitin0586_jpg.rf.8cad5fb756af5f7f315c920e6115014e.jpg)


---

## 🧠 Modelo entrenado

El modelo fue entrenado usando **YOLOv8** de Ultralytics como punto de partida, realizando un **fine-tuning** con este **[dataset](https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo)** ya etiquetado de **Roboflow** , posteriormente reducido y depurado por nosotros.

- **Modelo base:** `yolov8n.pt`
- **Framework:** PyTorch / Ultralytics YOLOv8
- **Formato exportado:** `.pt` (PyTorch) y `.onnx` (ONNX)
- **Número de clases detectadas:** 6  
