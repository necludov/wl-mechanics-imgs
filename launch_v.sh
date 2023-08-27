#!/bin/bash
# Node resource configurations
#SBATCH --job-name=train_jamot
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=4

# for normal t4v2,t4v1,a40
# for high t4v2
# for deadline t4v2,t4v1,a40
#SBATCH --partition=a40

#SBATCH --gres=gpu:4
#SBATC --account=deadline
#SBATCH --qos=normal
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

# Append is important because otherwise preemption resets the file
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

# creating dirs
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint/${SLURM_JOB_ID}
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE
mkdir /checkpoint/${USER}/${SLURM_JOB_ID}/checkpoints
# continue from
# cp /checkpoint/kirill/9003523/checkpoints/checkpoint_20 /checkpoint/${USER}/${SLURM_JOB_ID}/checkpoints/

source /ssd003/home/${USER}/.bashrc
source /ssd003/home/${USER}/venvs/jax-env/bin/activate


python main.py --config configs/mnist_ot_gen.py \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode 'train'

# python main.py --config configs/am/cifar/generation.py \
#                --workdir $PWD/checkpoint/8915516 \
#                --mode 'eval'

# python main.py --config configs/am/cifar/generation.py \
#                --workdir ~/jam \
#                --mode 'fid_stats'

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./logs/slurm-$SLURM_JOB_ID.out $archive/job.out
cp ./logs/slurm-$SLURM_JOB_ID.err $archive/job.err
