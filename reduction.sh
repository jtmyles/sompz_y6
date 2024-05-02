source ~/.bash_profile
conda activate sompz

CFGFILE=$1 # cfg/hsc_sompz_2024_05_01.cfg

#echo 'Begin training Deep SOM'
#ipython ~/repositories/sompz_y6/train_SOM_deep.py $CFGFILE

echo 'Begin training Wide SOM'
echo $(date)
ipython ~/repositories/sompz_y6/train_SOM_wide.py $CFGFILE

#echo 'Begin assignment of Balrog sample to SOMs'
#echo $(date)
#mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_deep_balrog_mpi.py $CFGFILE
#mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_wide_balrog_mpi.py $CFGFILE

echo 'Begin assignment of Wide sample to SOMs'
echo $(date)
mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_wide_noshear_hsc_mpi.py $CFGFILE
ipython ~/repositories/sompz_y6/utils/join_cell_assign_files.py $CFGFILE
