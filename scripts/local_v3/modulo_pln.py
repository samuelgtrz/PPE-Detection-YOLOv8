# pln.py
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PyPDF2 import PdfReader

# Importamos la librería de Google GenAI
from google import genai
from google.genai import types

# --- CONFIGURACIÓN ---
# Asegúrate de tener la variable de entorno GEMINI_API_KEY configurada en tu sistema.
# Si no, puedes descomentar la línea de abajo y ponerla (no recomendado por seguridad).
# os.environ["GEMINI_API_KEY"] = "TU_API_KEY_AQUI"

CARPETA_JSON = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Proyecto_Integrador_2\\Proyecto\\scripts\\local_v3\\data\\json"

# Inicializamos el cliente. Buscará automáticamente la variable de entorno 'GEMINI_API_KEY'
try:
    client = genai.Client()
except Exception as e:
    print(f"Error al inicializar cliente Gemini: {e}")
    print("Asegúrate de tener la variable de entorno GEMINI_API_KEY configurada.")
    sys.exit(1)


# --- UI ---

# Dentro de modulo_pln.py

# Dentro de modulo_pln.py

def preguntar_si_cargar_normativa():
    root = tk.Tk()
    root.withdraw() 
    
    try:
        # Enfoque agresivo para evitar que la ventana se manifieste
        root.update_idletasks()
        root.overrideredirect(True) # Oculta la barra de título de la ventana
        root.resizable(False, False)
        root.wm_attributes("-topmost", True) # La mantiene encima
        
        # Llama al cuadro de mensaje
        respuesta = messagebox.askyesno(
            "Normativa",
            "¿Quieres cargar una nueva normativa desde un PDF?"
        )
    finally:
        # ¡IMPORTANTE! Destrucción total garantizada
        root.destroy()
        
    return respuesta


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


# --- PROCESAMIENTO PDF → TEXTO ---

def extraer_texto_pdf(ruta_pdf):
    try:
        reader = PdfReader(ruta_pdf)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()
        return texto
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return ""


# --- PROCESAMIENTO LLM → JSON ---

def extraer_requisitos(texto_pdf):
    """
    Envía el texto a Gemini para extraer requisitos en formato JSON.
    """

    # Prompt para enviar a Gemini
    prompt = f"""
    The following text has been extracted from a PDF document.
    **Task:** Extract the PPE (EPI) requirements for each area and task in JSON format.

    **JSON Structure:** - The JSON must include:
        1. `document_id`: Extract the document ID from the text (or create a placeholder if missing).
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

    - Any type of helmet -> "casco de seguridad"
    - Any type of goggles or ocular protection -> "gafas de seguridad"
    - Any type of gloves -> "guantes de proteccion"
    - Any protective footwear -> "botas"
    - Any reflective vest -> "chaleco reflectante"

    If something does not match any category, IGNORE it and do NOT include it in the JSON.

    **Input Text:** ---
    {texto_pdf}
    ---

    **Important:** - The output must be in **Spanish**. 
    - For each `requisitos_epi`, the maximum number of PPEs with the same `nombre_epi` is 1.
    - Consider only `uso_obligatorio`: false if it's explicitly mentioned that is not obligatory or if it's optional.
    """

    try:
        # Llamada a la API de Gemini
        # Usamos 'gemini-2.0-flash' (rápido y eficiente) o 'gemini-1.5-flash'.
        # Configuramos response_mime_type para forzar JSON válido.
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        return response.text

    except Exception as e:
        print(f"Error en la llamada a Gemini: {e}")
        return "{}" # Retornar JSON vacío en caso de error para no romper el flujo


def guardar_json(json_data, ruta_pdf, carpeta_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida) # Crear carpeta si no existe

    nombre_sin_ext = os.path.splitext(os.path.basename(ruta_pdf))[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_sin_ext}.json")

    with open(ruta_salida, 'w', encoding='utf-8') as fichero:
        json.dump(json_data, fichero, ensure_ascii=False, indent=4)

    print("JSON guardado en:", ruta_salida)


# --- FUNCIÓN PRINCIPAL DEL PLN ---

def obtener_json_normativa():

    # Preguntar al usuario si quiere usar PDF o JSON
    usar_pdf = preguntar_si_cargar_normativa()

    if usar_pdf:
        # FLUJO PDF → JSON
        ruta_pdf = seleccionar_pdf()
        if not ruta_pdf:
            print("No se seleccionó PDF.")
            sys.exit()

        print(f"Procesando PDF: {ruta_pdf} ...")
        texto_pdf = extraer_texto_pdf(ruta_pdf)
        
        if not texto_pdf:
            print("El PDF no contiene texto extraíble.")
            sys.exit()

        json_string = extraer_requisitos(texto_pdf)

        # Limpieza robusta del JSON
        # A veces los modelos envuelven la respuesta en ```json ... ```
        # Como usamos response_mime_type='application/json', normalmente viene limpio,
        # pero prevenimos errores de markdown.
        clean_json_str = json_string.strip()
        if clean_json_str.startswith("```json"):
            clean_json_str = clean_json_str[7:]
        if clean_json_str.startswith("```"):
            clean_json_str = clean_json_str[3:]
        if clean_json_str.endswith("```"):
            clean_json_str = clean_json_str[:-3]
        
        try:
            json_data = json.loads(clean_json_str)
            if isinstance(json_data, list) and len(json_data) == 1:
                json_data = json_data[0] # Extraemos el diccionario del interior de la lista, esto lo hacemos porque a veces gemini devuelve mal el json y lo devuelve dentro de una lista
            guardar_json(json_data, ruta_pdf, CARPETA_JSON)
        except json.JSONDecodeError as e:
            print("Error al decodificar el JSON devuelto por Gemini.")
            print("Respuesta recibida:", json_string)
            sys.exit(1)

    else:
        # FLUJO JSON EXISTENTE
        ruta_json = seleccionar_json_existente()
        if not ruta_json:
            print("No se seleccionó JSON.")
            sys.exit()

        with open(ruta_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list) and len(json_data) == 1:
            json_data = json_data[0]

    return json_data

if __name__ == "__main__":
    # Bloque para probar el script independientemente
    try:
        datos = obtener_json_normativa()
        # print(json.dumps(datos, indent=2, ensure_ascii=False))
        print("Proceso finalizado con éxito.")
    except Exception as e:
        print(f"Error inesperado: {e}")