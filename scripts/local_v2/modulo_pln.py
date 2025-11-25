from openai import OpenAI
from PyPDF2 import PdfReader
import json
import tkinter as tk
from tkinter import filedialog
import os

CARPETA_JSON = "C:\\Users\\Roque\\Documents\\USC\\Curso_4\\Cuatri_1\\Proyecto_Integrador\\Work\\Proyecto-Integrador-de-IA\\local\\data\\json"

# Se crea una instancia de la clase OpenAI
client = OpenAI()  # Se necesita una API Key válida para usar OpenAI



# Función para seleccionar un archivo PDF
def seleccionar_pdf():  # Abrir el gestor de archivos para seleccionar un archivo PDF
    """
    Abre el gestor de archivos para que el usuario seleccione un archivo PDF.
    Devuelve la ruta del archivo seleccionado.
    """
    # Crear una ventana oculta de tkinter
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    # Abrir el cuadro de diálogo para seleccionar archivo
    ruta_pdf = filedialog.askopenfilename(
        title="Seleccionar archivo PDF",
        filetypes=[("Archivos PDF", "*.pdf")]  # Filtrar solo archivos PDF
    )

    if ruta_pdf:
        print(f"Archivo seleccionado: {ruta_pdf}")
        print(f"Modulo de PLN en proceso...\n")
    else:
        print("No se selecciono ningun archivo.")

    return ruta_pdf



# Función para extraer texto de un PDF
def extraer_texto_pdf(ruta_pdf):
    reader = PdfReader(ruta_pdf)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text()
    return texto



# Función para procesar el texto con ChatGPT
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

    # Obtener nombre del PDF sin extensión
    nombre_sin_ext = os.path.splitext(os.path.basename(ruta_pdf))[0]

    # Construir ruta final del JSON
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_sin_ext}.json")

    # Guardar JSON
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


# Seleccionar el archivo PDF
ruta_pdf = seleccionar_pdf()

# Procesar el PDF y guardar el JSON
texto_pdf = extraer_texto_pdf(ruta_pdf)
json_string = extraer_requisitos(texto_pdf)

# Extraer el JSON del texto generado
json_normativas = '\n'.join(json_string.strip().splitlines()[1:-1])

# Convertir el texto JSON a un objeto Python
json_data = json.loads(json_normativas)

# Guardar el JSON en un archivo
guardar_json(json_data, ruta_pdf, CARPETA_JSON)