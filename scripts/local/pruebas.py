import csv
epi_mapping_json = {
        "casco de seguridad": 0,
        "chaleco reflectante": 1,
        "guantes de protección": 2,
        "gafas de seguridad": 3,
        "botas": 4,
    }
epi_mapping_csv = {
    "Helmet": 0,
    "Safety Vest": 1,
    "Gloves": 2,
    "Glasses": 3,
    "Safety Boot": 4,
}
# Umbrales en el orden del mapping
# [Helmet = 0, Safety Vest = 1, Gloves = 2, Glasses = 3, Safety Boot = 4]
epis_threshold = [0.8, 0.8, 0.8, 0.8, 0.8]

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


epis_obligatorios = extract_epis(json_data={}, area_seleccionada="", epi_mapping=epi_mapping_json)  # Ejemplo de llamada a la función


# Crear reverse mapping para poder imprimir nombres JSON
reverse_mapping_json = {v: k for k, v in epi_mapping_json.items()}

# Inicializar contadores SOLO para EPIs obligatorios
stats = {
    reverse_mapping_json[idx]: {"validas": 0, "no_validas": 0}
    for idx in epis_obligatorios
}

# -------------------------
# LEER PREDICCIONES DEL CSV
# -------------------------

with open('C:\\Users\\Roque\\Documents\\USC\\Curso_3\\Cuatri_1\\Proyecto\\Entrega_final\\desarrollo\\roque\\predicciones_inferencia.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        epi_csv = row["epi"].strip()
        confidence = float(row["confidence"])

        # Comprobar que el EPI detectado existe en el mapping del CSV
        if epi_csv not in epi_mapping_csv:
            continue

        idx = epi_mapping_csv[epi_csv]  # índice del EPI detectado

        # Solo analizar EPIs obligatorios
        if idx not in epis_obligatorios:
            continue

        threshold = epis_threshold[idx]

        # Obtener el nombre en formato JSON para guardar el resultado
        name_json = reverse_mapping_json[idx]

        # Contar detecciones válidas o no válidas
        if confidence >= threshold:
            stats[name_json]["validas"] += 1
        else:
            stats[name_json]["no_validas"] += 1

# -------------------------
# MOSTRAR RESULTADOS
# -------------------------

print("\nRESULTADOS DE EPI:\n")
for epi, conteos in stats.items():
    print(f"{epi.upper()}:")
    print(f"  Válidas:     {conteos['validas']}")
    print(f"  No válidas:  {conteos['no_validas']}\n")
