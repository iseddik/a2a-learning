#!/usr/bin/env sh
#SBATCH --partition=gpu80G
#SBATCH --job-name=start   
#SBATCH --output=%x-%j.out             
#SBATCH --error=%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=16             
#SBATCH --mem=128G
###SBATCH --mail-user=issam.seddik@cea.fr
#SBATCH --mail-type=ALL

module load python/3.13.1

source /home/users/iseddik/vLLM/myenv/bin/activate

# Set cache directories for huggingface
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets


echo $CUDA_VISIBLE_DEVICES

python3 run_vllm.py
