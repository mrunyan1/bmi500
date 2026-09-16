#!/bin/bash -l
#SBATCH --job-name=pbmc10k_fine
#SBATCH --partition=overflow
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -e
umask 077
cd "$SLURM_SUBMIT_DIR"
case "$(hostname -s)" in
    oddjobs*) echo "Submit this script with sbatch."; exit 1 ;;
esac

module load bmi/python-3.12.12
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="$PWD/results/cache_${SLURM_JOB_ID}/numba"
export MPLCONFIGDIR="$PWD/results/cache_${SLURM_JOB_ID}/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

trap 'echo "Interrupted: this run is incomplete."; exit 143' TERM
date
hostname
git rev-parse HEAD
python --version
python -m pip freeze

outdir="results/pbmc10k_fine_${SLURM_JOB_ID}"
python -u scanpy_pbmc.py \
    --data-dir data --data-set pbmc10k \
    --out-dir "$outdir" --num-threads 1 --profile-mode fine
