#!/bin/bash
#SBATCH --job-name=yolo_finetune
#SBATCH --output=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/logs/yolo_finetune_%j.out
#SBATCH --error=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/logs/yolo_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=breixo.brea@rai.usc.es


# --- Limpiar y cargar módulos ---
module --force purge
module --ignore-cache load cesga/system miniconda3/22.11.1-1

# --- Activar el entorno YOLO ---
source $(conda info --base)/etc/profile.d/conda.sh  # asegura que conda está inicializado
conda activate /mnt/netapp2/Store_uni/home/usc/cursos/curso1278/yolov8_env

# --- Información de entorno ---
echo "=== INICIO DEL JOB ${SLURM_JOB_ID} ==="
hostname
nvidia-smi
echo "======================================"

# --- Ejecutar script de fine-tuning de YOLOv8 ---
mkdir -p logs  # asegura que el directorio logs está creado
python fine_tune_yolov8.py > logs/yolo_train_${SLURM_JOB_ID}.log 2>&1

# --- Estado final de la GPU ---
echo "=== GPU STATUS POST-TRAINING ==="
nvidia-smi

echo "=== JOB FINALIZADO ==="
date

