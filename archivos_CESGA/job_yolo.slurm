#!/bin/bash
#SBATCH --job-name=yolo_finetune
#SBATCH --output=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/logs/yolo_finetune_%j.out
#SBATCH --error=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/logs/yolo_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=breixo.brea@rai.usc.es

echo "=== INICIO DEL JOB ${SLURM_JOB_ID} ==="
date
hostname
nvidia-smi
echo "====================================="

# Cargar entorno conda
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/netapp2/Store_uni/home/usc/cursos/curso1278/yolov8_env
echo "Activate env: $CONDA_DEFAULT_ENV"

# Ir al directorio del proyecto
cd /mnt/netapp2/Store_uni/home/usc/cursos/curso1278 || exit 1

# Ejecutar script
python fine_tune_yolov8.py > logs/yolo_train_${SLURM_JOB_ID}.log 2>&1

echo "=== JOB FINALIZADO ==="
date

