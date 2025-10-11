#! /bin/bash
# ===== SLURM OPTIONS =====
#SBATCH --job-name="your jobname"
#SBATCH --partition=your_partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=./slurm_jobs/%j.out
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# ==== Main ======
# load your conda environment here, load any modules you need, etc.
echo $(nvidia-smi)

# run training scripts sequentially on the allocated nodes
# each script will use all nodes for distributed training
SCRIPT_PATHS=(
    "/path/to/ego_depth_video/train_viscop.sh"
    "/path/to/robotic_control/train_viscop.sh"
)

# get address of the master node (rank 0)
MASTER_ADDR_PASSED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) 

NUM_GPU_PER_NODE=1

echo started training runs

for i in "${!SCRIPT_PATHS[@]}"; do
    SCRIPT_PATH=${SCRIPT_PATHS[$i]}
    srun --export=ALL,MASTER_ADDR_PASSED="$MASTER_ADDR_PASSED" \
        bash $SCRIPT_PATH $SLURM_NNODES $NUM_GPU_PER_NODE
done

echo finished training runs