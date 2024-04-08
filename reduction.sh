source ~/.bash_profile
conda activate sompz

ipython ~/repositories/sompz_y6/train_SOM_deep.py hsc_sompz.cfg
ipython ~/repositories/sompz_y6/train_SOM_wide.py hsc_sompz.cfg
mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_deep_balrog_mpi.py hsc_sompz.cfg
mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_wide_balrog_mpi.py hsc_sompz.cfg
mpiexec -n 8 python ~/repositories/sompz_y6/assign_SOM_wide_noshear_hsc_mpi.py hsc_sompz.cfg
ipython ~/repositories/sompz_y6/utils/join_cell_assign_files.py hsc_sompz.cfg
