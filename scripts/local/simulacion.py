import subprocess
import os
from modulo_pln import json_data, ruta_pdf
import tkinter as tk
from tkinter import filedialog
import sys
import csv


def seleccionar_area(json_data):
    # Extraer las áreas del JSON
    areas = []
    for area in json_data.get("areas", []):
        area = area.get("nombre_area")
        areas.append(area)
    if areas == []:
        print("No se han encontrado áreas en el JSON.")
        sys.exit()
    
    # Crear una ventana de Tkinter
    root = tk.Tk()
    root.geometry("400x400+660+260")  # Establecer el tamaño de la ventana

    # Función para obtener el área seleccionada y cerrar la ventana
    def obtener_area():
        area_seleccionada = variable_area.get()  # Guardar el valor seleccionado en una variable
        print(f"Área seleccionada: {area_seleccionada}")   # Mostrar el valor en la consola
        root.quit()  # Salimos del root.mainloop(), pero no cierra la ventana
        return area_seleccionada

    # Crear un StringVar para mantener la selección del área
    variable_area = tk.StringVar(root)
    variable_area.set(areas[0])  # Establecer el valor predeterminado

    # Crear el menú desplegable (OptionMenu) para seleccionar el área
    menu_area = tk.OptionMenu(root, variable_area, *areas)

    # Aumentar el tamaño de la fuente del menú
    menu_area.config(font=("Arial", 14))  # Cambiar el tamaño de la fuente del menú

    # Aumentar el tamaño de la opción seleccionada (anchura y altura)
    menu_area.config(width=40)  # Establecer el ancho del menú
    menu_area.config(height=4)  # Establecer la altura de las opciones
    menu_area.pack(pady=10)  # Añadir un poco de espacio alrededor del menú

    # Botón para obtener el área seleccionada y cerrar la ventana
    boton_seleccionar = tk.Button(
        root,
        text="Seleccionar área",
        command=lambda: obtener_area(),
        width=20,  # Aumentar el ancho
        height=3,  # Aumentar la altura
        font=("Arial", 14),  # Aumentar el tamaño de la fuente
        bg="lightblue",  # Cambiar el color de fondo del botón
        fg="black"  # Cambiar el color del texto
    )
    boton_seleccionar.pack(pady=20)  # Añadir un poco de espacio alrededor del botón

    # Iniciar el bucle principal de la ventana
    root.mainloop()

    area_seleccionada = variable_area.get()  # Obtener el valor seleccionado
    root.destroy()  # Cerrar la ventana y eliminar todos los recursos de root

    return area_seleccionada



epi_mapping_json = {
        "casco de seguridad": ["Helmet",0.8], # [Name, Threshold]
        "chaleco reflectante": ["Safety Vest",0.8],
        "guantes de protección": ["Gloves",0.8],
        "gafas de seguridad": ["Glasses",0.8],
        "botas": ["Safety Boot",0.8],
    }

# Función para extraer los EPIs de un JSON
def extract_epis(json_data, area_seleccionada, epi_mapping):
    epis = set()

    # Convertir nombres comunes de EPIs a las palabras clave del prompt
    

    # Extraer EPIs de las áreas
    for area in json_data.get("areas", []):
        if area.get("nombre_area", [])==area_seleccionada:
            for epi in area.get("requisitos_epi", []):
                if epi.get("uso_obligatorio", False):  # 
                    epis.add(epi.get("nombre_epi", "").lower()) 
            return [epi_mapping[epi] for epi in epis if epi in epi_mapping]

    print("No se han encontrado EPIs para el área seleccionada.")
    return []


def seleccionar_imagen(): # abrir el gestor de archivos para seleccionar una imagen
    """
    Abre el gestor de archivos para que el usuario seleccione una archivo imagen.
    Devuelve la ruta del archivo seleccionado.
    """
    # Crear una ventana oculta de tkinter
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    # Abrir el cuadro de diálogo para seleccionar archivo
    ruta_imagen = filedialog.askopenfilename(
        title="Seleccionar una imagen jpg",
        filetypes=[("Imagenes jpg", "*.jpg")]  # Filtrar solo archivos PDF
    )

    if ruta_imagen:
        print(f"\nArchivo seleccionado: {ruta_imagen}")
    else:
        print("No se seleccionó ningún archivo.")

    return ruta_imagen

def verificar_epis_en_csv(csv_path, epis_obligatorios):
    """
    epis_obligatorios = lista de listas: [nombre_epi_modelo, threshold]
    Ej: ["Helmet", 0.8]
    """

    # Convertimos formato:
    # dict con clave = nombre modelo → threshold
    epis_dict = {epi[0].lower(): epi[1] for epi in epis_obligatorios}

    encontrados = 0
    no_encontrados = 0
    detalles = []

    # Leemos CSV
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            nombre = row["clase_nombre"].lower()
            confianza = float(row["confianza"])

            if nombre in epis_dict:
                threshold = epis_dict[nombre]
                supera = confianza >= threshold

                if supera:
                    encontrados += 1
                else:
                    no_encontrados += 1

                detalles.append({
                    "epi": nombre,
                    "confianza": confianza,
                    "threshold": threshold,
                    "supera_threshold": supera
                })

    return encontrados, no_encontrados, detalles


def main():
    # Elegir área
    area_seleccionada = seleccionar_area(json_data)

    # Extraer EPIs
    epis_obligatorios = extract_epis(json_data=json_data, area_seleccionada=area_seleccionada, epi_mapping=epi_mapping_json)  # Ejemplo de llamada a la función

    if epis_obligatorios == []:
        print("No se han encontrado EPIs para el área introducida.")

    else:
        csv_path = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\roque\\predicciones_inferencia.csv"

        encontrados, no_encontrados, detalles = verificar_epis_en_csv(csv_path, epis_obligatorios)

        print("EPIs obligatorios que SUPERAN el threshold:", encontrados)
        print("EPIs obligatorios que NO superan el threshold:", no_encontrados)

        print("\nDetalles:")
        for d in detalles:
            print(d)


if __name__ == "__main__":
    main()
