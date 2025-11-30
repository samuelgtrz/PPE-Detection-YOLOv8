import cv2
from PIL import Image

ruta = r"C:\\Users\\Roque\\Documents\\casco_chaleco_gafas.jpg"

print("Probando con OpenCV...")
img_cv = cv2.imread(ruta)
print("Resultado OpenCV:", "OK" if img_cv is not None else "ERROR")

print("\nProbando con PIL...")
try:
    img_pil = Image.open(ruta)
    img_pil.verify()
    print("Resultado PIL: OK")
except Exception as e:
    print("Resultado PIL: ERROR -", e)
