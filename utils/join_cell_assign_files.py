import os
import sys
import yaml
import h5py
import pickle
import numpy as np

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

if len(sys.argv) > 2:
    nproc_balrog_data = sys.argv[2] # number of mpi processes/cores used to run assign script
    nproc_wide_data = sys.argv[2] # number of mpi processes/cores used to run assign script
    nsubsets = sys.argv[2] # should match variable in assign script
else
    nproc_balrog_data = 16 
    nproc_wide_data = 16 
    nsubsets = 16 

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

som_type = 'wide'
output_path = cfg['out_dir']
som_wide = cfg['som_wide']
som_dim = cfg['wide_som_dim']
run_name = cfg['run_name']
shear = 'noshear'
path_out = os.path.join(output_path, f'{som_type}_{shear}')

nTrain = cfg['nwide_train']
#suffix = int(np.log10(nTrain))
train_suffix = f'{nTrain:.0e}'.replace('+0','')
def join_cell_assign_files(path_cats, som_type, shear, run_name, nproc):
    cells = []
    for item in range(nproc):
        for subsample in range(nsubsets):
            # load file
            if som_type == 'wide':
                cells_subsample = np.load(
                    f'{path_cats}/{som_type}_{shear}/som_wide_{som_dim}_{som_dim}_{train_suffix}_{run_name}_assign_{shear}_{item}_subsample_{subsample}.npz')[
                    'cells']
            else:
                cells_subsample = np.load(
                    f'{path_cats}/{som_type}_{shear}/som_deep_assign_{shear}_{item}_subsample_{subsample}.npz')[
                    'cells']
            # append cells
            cells.append(cells_subsample)
            print(f'Found {len(cells_subsample)} in file.')
    cells = np.concatenate(cells)
    ncells = len(cells)
    print(f'{ncells} cells in total')
    with open(f"{path_cats}/cells_{som_type}_{shear}_{run_name}.pkl", "wb") as output_file:
        pickle.dump(cells, output_file)

    # with h5py.File('sompz.hdf5', 'w', track_order=True) as f:
    #     f.create_dataset(f'catalog/sompz/{shear}/cell_wide', data=cells)
    # print(f'done with {shear}')


#path_cats = '/global/cfs/projectdirs/des/acampos/sompz_output/y6_data_preliminary'
#run_name = 'y6preliminary'
#path_cats = '/Users/jmyles/data/sompz_desc/2024-01-03'
path_cats = output_path

join_cell_assign_files(path_cats, 'deep', 'deep_balrog', run_name, nproc_balrog_data)
join_cell_assign_files(path_cats, 'wide', 'deep_balrog', run_name, nproc_balrog_data)
join_cell_assign_files(path_cats, 'wide', 'noshear', run_name, nproc_wide_data)

# join_cell_assign_files(path_cats, som_type, 'sheared_1m', run_name)
# join_cell_assign_files(path_cats, som_type, 'sheared_1p', run_name)
# join_cell_assign_files(path_cats, som_type, 'sheared_2m', run_name)
# join_cell_assign_files(path_cats, som_type, 'sheared_2p', run_name)
