import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json

# Función para reescalar la imagen
def resize_image(image_path, max_size=(512, 512)):
    """
    Reescala una imagen para que no supere el tamaño máximo dado.

    Args:
        image_path (str): Ruta al archivo de imagen.
        max_size (tuple): Dimensiones máximas (ancho, alto) para la imagen.

    Returns:
        Image: Imagen reescalada como objeto PIL.Image.
    """
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.ANTIALIAS)  # Reescala manteniendo la relación de aspecto
    return image

def leer_txt(file_path):  # Puede leer el prompt o la ruta de salida, reutilizamos la función
    with open(file_path, "r") as file:
        contenido = file.read()
    return contenido

def generar_respuesta(all_true, output_text):  # Se pasa el booleano y la salida del modelo
    if all_true:  # Crear un txt y guardarlo
        with open("respuesta.txt", "w") as file:
            file.write("Todos los EPIs necesarios en esta área están presentes y colocados correctamente, el trabajador puede acceder. \nPara más información, revise el log completo")
    else:
        with open("respuesta.txt", "w") as file:
            file.write("Faltan EPIs o no están correctamente colocados, acceso denegado, generando alerta. \nPara más información, revise el log completo")

    # Mostrar la salida del modelo
    with open("respuesta.txt", "a") as file:
        file.write("\n\nEl modelo ha generado la siguiente salida:\n")
        file.write(output_text)

# Ruta a la imagen que quieres procesar, revisar para ver si  se puede hacer un menu
image_path = "/mnt/netapp2/Store_uni/home/usc/cursos/cursoa83/fotos/prueba.jpg"

# Reescalar la imagen antes de procesarla
image_resized = resize_image(image_path, max_size=(512, 512))

# Configurar el modelo para usar float16 y offloading
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float32,  # Usa float16 para reducir memoria
    device_map="auto",  # Divide automáticamente entre GPU y RAM
    attn_implementation="sdpa"  # Activa el flash attention pip install flash-attn --no-build-isolation

)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct"
)

# Definir mensajes con imagen y texto
prompt = leer_txt("prompt_vlm.txt")
print("\nEl prompt utilizado es el siguiente: \n", prompt)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_resized,  # Usamos la imagen reescalada
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparar el texto e imágenes para el modelo
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Mover datos a GPU si es necesario
inputs = inputs.to("cuda")

# Liberar memoria antes de la generación (opcional)
torch.cuda.empty_cache()

# Generación con menos tokens nuevos para evitar OOM
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Recortar los IDs generados para eliminar entrada original
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

#Decodificar la salida en texto
output = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Mostrar el resultado
print("\nEl modelo ha generado la siguiente salida:\n", output)

# Transformamos la salida
output_text = str(output)

# Paso 1: Limpiar la cadena inicial (eliminar corchetes y saltos de línea)
cleaned_str = output_text.strip("[]").strip("'").replace("\\n", "")

# Paso 2: Reemplazar "false" y "true" por false y true (JSON válido)
formatted_str = cleaned_str.replace('"false"', 'false').replace('"true"', 'true')

# Paso 3: Usar json.loads para convertirlo en un objeto Python
epi_diccionario = json.loads(formatted_str)

# Comprobamos si todos los EPIs están presentes
booleano_epis_presentes = all(epi_diccionario.values())
generar_respuesta(booleano_epis_presentes, output_text)
