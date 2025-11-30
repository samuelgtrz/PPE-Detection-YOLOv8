# pln.py
from openai import OpenAI
from PyPDF2 import PdfReader
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

CARPETA_JSON = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\scripts\\local_v2\\data\\json"

client = OpenAI()


# UI


def preguntar_si_cargar_normativa():
    root = tk.Tk()
    root.withdraw()
    return messagebox.askyesno(
        "Normativa",
        "¿Quieres cargar una nueva normativa desde un PDF?"
    )


def seleccionar_pdf():
    root = tk.Tk()
    root.withdraw()
    ruta_pdf = filedialog.askopenfilename(
        title="Seleccionar archivo PDF",
        filetypes=[("Archivos PDF", "*.pdf")]
    )
    return ruta_pdf


def seleccionar_json_existente():
    root = tk.Tk()
    root.withdraw()
    ruta_json = filedialog.askopenfilename(
        title="Seleccionar archivo JSON",
        filetypes=[("Archivos JSON", "*.json")]
    )
    return ruta_json



# PROCESAMIENTO PDF → TEXTO


def extraer_texto_pdf(ruta_pdf):
    reader = PdfReader(ruta_pdf)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text()
    return texto



# PROCESAMIENTO LLM → JSON


def extraer_requisitos(texto_pdf):

    # Prompt para enviar a ChatGPT
    prompt = f"""
    The following text has been extracted from a PDF document.
    **Task:** Extract the PPE (EPI) requirements for each area and task in JSON format.

    **JSON Structure:**  
    - The JSON must include:
        1. `document_id`: Extract the document ID from the text.
        2. `titulo`: Set a title for the document, starting with "Normativa de Seguridad Industrial - ".
        3. `fecha`: Extract the date from the text in "DD-MM-YYYY" format.
        4. `areas`: A list of areas, each with:
            - `nombre_area`: Name of the area.
            - `requisitos_epi`: A list of required PPEs for the area, each with:
                - `nombre_epi`: Name of the PPE.
                - `uso_obligatorio`: Boolean value indicating mandatory usage.
        5. `tareas`: A list of tasks, each with:
            - `nombre_tarea`: Name of the task.
            - `requisitos_epi`: A list of required PPEs for the task, with the same structure as above.

    Replace all placeholders with appropriate values in Spanish.

    IMPORTANT:  
    You must use ONLY the following valid PPE names (`nombre_epi`).  
    No other PPE names are allowed in the JSON output:

    - "casco de seguridad"
    - "chaleco reflectante"
    - "guantes de proteccion"
    - "gafas de seguridad"
    - "botas"

    If the text mentions a different PPE (e.g., “calzado de seguridad”, “protección ocular”, “casco de soldador”, “guantes de soldador”, “auriculares”, “máscara”, etc.), you MUST convert it to the closest valid option above:

    - Any type of helmet → "casco de seguridad"
    - Any type of goggles or ocular protection → "gafas de seguridad"
    - Any type of gloves → "guantes de protección"
    - Any protective footwear → "botas"
    - Any reflective vest → "chaleco reflectante"

    If something does not match any category, IGNORE it and do NOT include it in the JSON.


    **Input Text:**  
    ---
    {texto_pdf}
    ---

    **Important:**  
    - Return **only** the JSON output, without any comments or explanations.  
    - The output must be in **Spanish**. 
    - For each `requisitos_epi`, the maximum number of PPEs with the same `nombre_epi` is 1.
    - Consider only `uso_obligatorio`: false if it's explicitly mentioned that is not obligatory or if it's optional.
    """

    # Llamada a la API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a useful assistant that extracts structured data."},  # "role": "system" indica el texto del asistente
            {"role": "user", "content": prompt}  # "role": "user" indica el texto del usuario
        ]
    )

    # Devuelve la respuesta de ChatGPT
    return response.choices[0].message.content

def guardar_json(json_data, ruta_pdf, carpeta_salida):
    nombre_sin_ext = os.path.splitext(os.path.basename(ruta_pdf))[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_sin_ext}.json")

    with open(ruta_salida, 'w', encoding='utf-8') as fichero:
        json.dump(json_data, fichero, ensure_ascii=False, indent=4)

    print("JSON guardado en:", ruta_salida)



# Ejemplo de JSON de salida esperado
"""
**Example JSON Output:**  
{
    "document_id": 12345,
    "titulo": "Normativa de Seguridad Industrial - documento nro. 15",
    "fecha": "29-11-2024",
    "areas": [
        {
            "nombre_area": "Planta de Producción",
            "requisitos_epi": [
                {
                    "nombre_epi": "casco de seguridad",
                    "uso_obligatorio": true
                },
                {
                    "nombre_epi": "guantes de protección",
                    "uso_obligatorio": true
                }
            ]
        },
        {
            "nombre_area": "Planta de Embalaje",
            "requisitos_epi": [
                {
                    "nombre_epi": "guantes de protección",
                    "uso_obligatorio": true
                },
            ]
        }
    ],
    "tareas": [
        {
            "nombre_tarea": "Mantenimiento de maquinaria",
            "requisitos_epi": [
                {
                    "nombre_epi": "auriculares",
                    "uso_obligatorio": true
                }
            ]
        }
    ]
}
"""


# FUNCIÓN PRINCIPAL DEL PLN


def obtener_json_normativa():

    # Preguntar al usuario si quiere usar PDF o JSON
    usar_pdf = preguntar_si_cargar_normativa()

    if usar_pdf:
        # FLUJO PDF → JSON
        ruta_pdf = seleccionar_pdf()
        if not ruta_pdf:
            print("No se seleccionó PDF.")
            sys.exit()

        texto_pdf = extraer_texto_pdf(ruta_pdf)
        json_string = extraer_requisitos(texto_pdf)

        # Limpiar respuesta del modelo
        json_normativas = '\n'.join(json_string.strip().splitlines()[1:-1])
        json_data = json.loads(json_normativas)

        guardar_json(json_data, ruta_pdf, CARPETA_JSON)

    else:
        # FLUJO JSON EXISTENTE
        ruta_json = seleccionar_json_existente()
        if not ruta_json:
            print("No se seleccionó JSON.")
            sys.exit()

        with open(ruta_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)

    return json_data
