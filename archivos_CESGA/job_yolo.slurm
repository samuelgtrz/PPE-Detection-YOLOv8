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
module purge
module --ignore-cache load cesga/system miniconda3/22.11.1-1

# --- Activar el entorno YOLO ---
conda activate /mnt/netapp2/Store_uni/home/usc/cursos/curso1278/yolov8_env

echo "=== INICIO DEL JOB ${SLURM_JOB_ID} ==="
date
hostname
nvidia-smi
echo "======================================"

# --- Ejecutar script de fine-tuning de YOLOv8 ---
python fine_tune_yolov8.py > logs/yolo_train_${SLURM_JOB_ID}.log 2>&1

echo "=== JOB FINALIZADO ==="
date

