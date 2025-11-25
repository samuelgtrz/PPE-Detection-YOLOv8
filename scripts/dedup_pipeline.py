# dedup_pipeline.py
import os, hashlib
from PIL import Image
from tqdm import tqdm
import imagehash
import numpy as np
import faiss
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from collections import defaultdict

# ------------ CONFIG -------------
BASE_DIR = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\datasets\\6_dataset_reducido_etiquetas_bien"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

EMB_FILE = os.path.join(BASE_DIR, "embeddings.npy")
FILES_FILE = os.path.join(BASE_DIR, "files.npy")
REPORT_FILE = os.path.join(BASE_DIR, "dup_report.txt")

PHASH_THRESH = 8           # Hamming threshold for pHash
COSINE_DUP_THRESH = 0.95   # threshold for embeddings similarity
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------

# 1) Recolectar imágenes
def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [os.path.join(root, f) for f in os.listdir(root)
             if os.path.splitext(f)[1].lower() in exts]
    return sorted(files)

files = list_images(IMG_DIR)
print(f"Found {len(files)} images")

# 2) Duplicados exactos (SHA256)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

sha_map = {}
dups_exact = []
for p in tqdm(files, desc="SHA256"):
    s = sha256(p)
    if s in sha_map:
        dups_exact.append((sha_map[s], p))
    else:
        sha_map[s] = p

print("Exact duplicates found:", len(dups_exact))

unique_files = list(sha_map.values())

# 3) Duplicados similares por pHash
phash_map = {}
phash_groups = defaultdict(list)
for p in tqdm(unique_files, desc="pHash"):
    try:
        img = Image.open(p).convert("RGB")
        h = imagehash.phash(img)
        phash_map[p] = h
    except Exception as e:
        print("Error:", p, e)

# Agrupar imágenes con pHash similar
for i, pa in enumerate(unique_files):
    for pb in unique_files[i + 1:]:
        if abs(phash_map[pa] - phash_map[pb]) <= PHASH_THRESH:
            phash_groups[pa].append(pb)

# 4) Extracción de embeddings (ResNet50)
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(DEVICE)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embeddings(file_list):
    embs = np.zeros((len(file_list), 2048), dtype="float32")
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_files:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)
            out = out.cpu().numpy()
            embs[i:i+len(batch_files)] = out
    # Normalizar embeddings (L2)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs

embeddings = extract_embeddings(unique_files)
np.save(EMB_FILE, embeddings)
np.save(FILES_FILE, np.array(unique_files))
print("Embeddings guardados en:", EMB_FILE)

# 5) FAISS: búsqueda por similitud coseno
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
k = 5
D, I = index.search(embeddings, k)

# 6) Construir pares de similitud alta
pairs = []
for i in range(len(unique_files)):
    for j_idx, score in zip(I[i, 1:], D[i, 1:]):  # saltar self
        if score >= COSINE_DUP_THRESH:
            pairs.append((i, int(j_idx), float(score)))

print("Pairs above threshold:", len(pairs))

# 7) Agrupar duplicados (Union-Find)
parent = list(range(len(unique_files)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for a, b, _ in pairs:
    union(a, b)

groups = defaultdict(list)
for i in range(len(unique_files)):
    groups[find(i)].append(i)

dupe_groups = [[unique_files[idx] for idx in ids] for ids in groups.values() if len(ids) > 1]
print(f"Found {len(dupe_groups)} duplicate groups (len>1)")

# 8) Guardar informe
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    for g in dupe_groups:
        f.write("GROUP:\n")
        for p in g:
            f.write(p + "\n")
        f.write("\n")

print(f"Informe guardado en {REPORT_FILE}")
