# simulacion.py
import subprocess
import os
import tkinter as tk
from tkinter import filedialog
import sys
import csv
from modulo_pln import obtener_json_normativa
from modulo_vc import model_loader, execute_model


MODE_PATH = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\resultados_ejecuciones\\5_comparacion_runs_modelos\\detect_m_no_freeze\\ppe_yolov8_finetuned\\weights\\best.pt"
OUTPUT_CSV_BASE = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\scripts\\local_v2\\data\\csv"
#IMAGES_PATH = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\scripts\\local_v2\\data\\fotos_pruebas"
IMAGES_PATH = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\scripts\\local_v2\\data\\test_imagtes\\images"
epi_mapping_json = {
        "casco de seguridad": ["Helmet",0.8],
        "chaleco reflectante": ["Safety Vest",0.8],
        "guantes de proteccion": ["Gloves",0.8],
        "gafas de seguridad": ["Glasses",0.8],
        "botas": ["Safety Boot",0.8],
    }


def seleccionar_area(json_data):

    areas = [a["nombre_area"] for a in json_data.get("areas", [])]

    if not areas:
        print("No hay áreas en el JSON.")
        sys.exit()

    root = tk.Tk()
    root.geometry("400x400+660+260")

    variable_area = tk.StringVar(root)
    variable_area.set(areas[0])

    menu_area = tk.OptionMenu(root, variable_area, *areas)
    menu_area.config(font=("Arial",14), width=40, height=3)
    menu_area.pack(pady=10)

    tk.Button(root, text="Seleccionar área",
              command=root.quit,
              width=20, height=3,
              font=("Arial",14),
              bg="lightblue").pack(pady=20)

    root.mainloop()
    area_sel = variable_area.get()
    root.destroy()
    return area_sel


def extract_epis(json_data, area_sel, epi_mapping):
    epis = []
    for area in json_data.get("areas", []):
        if area["nombre_area"] == area_sel:
            for epi in area.get("requisitos_epi", []):
                if epi.get("uso_obligatorio", False):
                    nombre = epi["nombre_epi"].lower()
                    if nombre in epi_mapping:
                        epis.append(epi_mapping[nombre])
    return epis


def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen jpg",
        filetypes=[("Imagen JPG", "*.jpg")]
    )
    return ruta


def verificar_epis_en_csv(csv_path, epis_obligatorios):

    epis_dict = {epi[0].lower(): epi[1] for epi in epis_obligatorios}

    encontrados = 0
    no_encontrados = 0
    detalles = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre = row["clase_nombre"].lower()
            conf = float(row["confianza"])

            if nombre in epis_dict:
                thr = epis_dict[nombre]
                supera = conf >= thr

                if supera:
                    encontrados += 1
                else:
                    no_encontrados += 1

                detalles.append({
                    "epi": nombre,
                    "confianza": conf,
                    "threshold": thr,
                    "supera_threshold": supera
                })

    return encontrados, no_encontrados, detalles


def ejecutar_simulacion(json_data):

    area_sel = seleccionar_area(json_data)
    epis_oblig = extract_epis(json_data, area_sel, epi_mapping_json)

    model = model_loader(MODE_PATH)
    # Procesar TODAS las imágenes de la carpeta IMAGES_PATH
    imagenes = [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(".jpg")]

    if not imagenes:
        print("No se encontraron imágenes en la carpeta.")
        sys.exit()

    for img in imagenes:
        ruta_imagen = os.path.join(IMAGES_PATH, img)

        nombre = os.path.splitext(img)[0]
        csv_out = os.path.join(OUTPUT_CSV_BASE, f"{nombre}.csv")

        print(f"\nProcesando imagen: {img}")

        execute_model(model, ruta_imagen, csv_out)

        encontrados, no_encontrados, detalles = verificar_epis_en_csv(csv_out, epis_oblig)

        print("EPIs que superan threshold:", encontrados)
        print("EPIs que NO superan threshold:", no_encontrados)
        print("Detalles:", detalles)
        print("-" * 50)



def main():
    json_data = obtener_json_normativa()   # <- PDF o JSON
    ejecutar_simulacion(json_data)         # <- Simulación YOLO

if __name__ == "__main__":
    main()