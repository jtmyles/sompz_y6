#!/bin/bash
#SBATCH --job-name=sompz         # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count. each node on della has 32 cores (i.e., cpus)
#SBATCH --ntasks-per-node=16     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jm8767@astro.princeton.edu

# load modules
#source /home/jm8767/.bash_profile
module purge
module load anaconda3/2024.2
module load openmpi/gcc/4.1.2
#module load intel/2022.2.0
#module load intel-mpi/intel/2021.7.0

source $HOME/.bash_profile
conda activate sompz

CFGFILE=$HOME/repositories/sompz_y6/cfg/hsc_sompz_2024_05_13.cfg # $1
NCORES=8

echo 'Begin training Deep SOM'
ipython $HOME/repositories/sompz_y6/train_SOM_deep.py $CFGFILE

echo 'Begin training Wide SOM'
echo $(date)
ipython $HOME/repositories/sompz_y6/train_SOM_wide.py $CFGFILE

echo 'Begin assignment of Balrog sample to SOMs'
echo $(date)
mpiexec -n $NCORES python $HOME/repositories/sompz_y6/assign_SOM_deep_balrog_mpi.py $CFGFILE
mpiexec -n $NCORES python $HOME/repositories/sompz_y6/assign_SOM_wide_balrog_mpi.py $CFGFILE
echo 'Begin assignment of Wide sample to SOMs'
echo $(date)
mpiexec -n $NCORES python $HOME/repositories/sompz_y6/assign_SOM_wide_noshear_hsc_mpi.py $CFGFILE

echo 'Combine info from assignment files'
ipython $HOME/repositories/sompz_y6/utils/join_cell_assign_files.py $CFGFILE $NCORES
echo $(date)
