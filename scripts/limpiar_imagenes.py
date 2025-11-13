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
IMG_DIR = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\imagenes_limpias"        # carpeta con imágenes (subcarpetas: train/ val/ test o plana)
EMB_FILE = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\embeddings.npy"
FILES_FILE = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Proxecto Integrador II\\Proyecto-Integrador-de-IA\\files.npy"
PHASH_THRESH = 8           # Hamming threshold for pHash
COSINE_DUP_THRESH = 0.95   # threshold for embeddings similarity
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------

# 1) Recolectar archivos
def list_images(root):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    files=[]
    for dirpath,_,fnames in os.walk(root):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(dirpath,f))
    return sorted(files)

files = list_images(IMG_DIR)
print(f"Found {len(files)} images")

# 2) Exact duplicate via SHA256
def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
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

# Optionally drop exact duplicates now (keep first occurrence)
unique_files = list(sha_map.values())

# 3) pHash filter
phash_map = {}
phash_groups = defaultdict(list)
for p in tqdm(unique_files, desc="pHash"):
    try:
        img = Image.open(p).convert("RGB")
        h = imagehash.phash(img)
        phash_map[p] = h
    except Exception as e:
        print("Error:", p, e)

# group by similar phash (cheap n^2 on reduced set; but 2000 is small)
for i,pa in enumerate(unique_files):
    for pb in unique_files[i+1:]:
        if abs(phash_map[pa] - phash_map[pb]) <= PHASH_THRESH:
            phash_groups[pa].append(pb)

# Build candidate set after pHash (keep unique_files but mark groups)
# 4) Embedding extraction (ResNet50)

# Load model with weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(DEVICE)

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_embeddings(file_list):
    embs = np.zeros((len(file_list), 2048), dtype='float32')
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_files:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)  # (B,2048)
            out = out.cpu().numpy()
            embs[i:i+len(batch_files)] = out
    # L2 normalize
    norms = np.linalg.norm( s, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embs = embs / norms
    return embs

# extract embeddings for unique files
embeddings = extract_embeddings(unique_files)
np.save(EMB_FILE, embeddings)
np.save(FILES_FILE, np.array(unique_files))

# 5) FAISS index (inner product because embeddings normalized -> cosine)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
k = 5
D, I = index.search(embeddings, k)  # D: scores (cosine), I: indices

# 6) Build pairs above threshold
pairs = []
for i in range(len(unique_files)):
    for j_idx,score in zip(I[i,1:], D[i,1:]):  # skip self (first result)
        if score >= COSINE_DUP_THRESH:
            pairs.append((i, int(j_idx), float(score)))

print("Pairs above threshold:", len(pairs))

# 7) Union-Find to create groups
parent = list(range(len(unique_files)))
def find(a):
    while parent[a]!=a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a,b):
    ra,rb = find(a), find(b)
    if ra!=rb:
        parent[rb] = ra

for a,b,_ in pairs:
    union(a,b)

groups = defaultdict(list)
for i in range(len(unique_files)):
    groups[find(i)].append(i)

# 8) Report groups
dupe_groups = [ [unique_files[idx] for idx in ids] for ids in groups.values() if len(ids)>1 ]
print(f"Found {len(dupe_groups)} duplicate groups (len>1)")

# Save or print small report
with open("dup_report.txt","w", encoding="utf-8") as f:
    for g in dupe_groups:
        f.write("GROUP:\n")
        for p in g:
            f.write(p + "\n")
        f.write("\n")

print("Report written to dup_report.txt")
