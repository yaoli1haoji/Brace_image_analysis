#!/bin/bash
#SBATCH --job-name=brace_root_machine_learning         # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=15
#SBATCH --ntasks=1
#SBATCH --mem=500gb                     # Job memory request
#SBATCH --time=36:00:00               # Time limit hrs:min:sec
#SBATCH --output=Unet3plus.%j.out    # Standard output log
#SBATCH --error=Unet3plus.%j.err     # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hl46161@uga.edu  # Where to send mail	

cd $SLURM_SUBMIT_DIR

#module load  Python/3.8.6-GCCcore-10.2.0

module load Miniconda3/4.10.3

module load CUDA/11.3.1

#module load CUDA/11.2.1-GCC-8.3.0

#module load NCCL/2.8.3-CUDA-11.1.1

export LD_LIBRARY_PATH=/home/hl46161/python3_conda/lib:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=/home/hl46161/.conda/envs/brace_root_environment/lib:$LD_LIBRARY_PATH

#pip install --user segmentation_models

eval "$(conda shell.bash hook)"

conda activate /home/hl46161/python3_conda

#conda  activate brace_root_environment

#conda activate instance_segmentation_env

TF_GPU_ALLOCATOR=cuda_malloc_async

#export CUDA_VISIBLE_DEVICES=0

#export NCCL_DEBUG=WARN

nvidia-smi

#python brace_root_machine_learning_unet3plusmore.py

python brace_root_machine_learning_unet3plus.py

#python brace_root_machine_learning_unet_origin_shape.py

#python brace_root_machine_learning_unet.py

#python brace_root_machine_learning_simple_unet.py

#python brace_root_machine_learning_resnet.py

#python brace_root_machine_learning_Attention_Unet.py

#python test.py


