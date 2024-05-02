import os
import sys
import numpy as np
import yaml
import h5py
from sompz import NoiseSOM as ns
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
print(f'MPI comm: {comm}. rank: {rank}. nprocs: {nprocs}')

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

som_type = 'wide'
# Read variables from config file
debug = cfg['debug']
nTrain = cfg['nTrain']
output_path = cfg['out_dir']
som_wide = cfg['som_wide']
som_dim = cfg['wide_som_dim']
bands = cfg['wide_bands']
#bands_label = cfg['wide_bands_label']
#bands_err_label = cfg['wide_bands_err_label']
#no_shear = cfg['shear_types'][0]
run_name = cfg['run_name']
catname = cfg['wide_file'] #'/Users/jmyles/data/hsc/sompz/2024-01-03/wide_data.h5'
shear = 'noshear'

path_out = os.path.join(output_path, f'{som_type}_{shear}')
if not os.path.exists(path_out):
    os.system(f'mkdir -p {path_out}')
som_wide = f'som_wide_{som_dim}_{som_dim}_1e7.npy'

if rank == 0:
    with h5py.File(catname, 'r') as f:
        fluxes = {}
        flux_errors = {}

        for i, band in enumerate(bands):
            print(i, band)
            if debug:
                # make slice with 100 random elements
                select = np.random.choice(len(f[f'/catalog/{shear}/flux_{band}']), 
                                          size=nTrain, replace=False)
            else:
                select = ...

            fluxes[band] = np.array_split(
                f[f'/catalog/{shear}/flux_{band}'][select],
                nprocs
            )

            flux_errors[band] = np.array_split(
                f[f'/catalog/{shear}/flux_err_{band}'][select],
                nprocs
            )

else:
    # data = None
    fluxes = {b: None for b in bands}
    flux_errors = {b: None for b in bands}

# scatter data
for i, band in enumerate(bands):
    fluxes[band] = comm.scatter(fluxes[band], root=0)
    flux_errors[band] = comm.scatter(flux_errors[band], root=0)

# prepare big data matrix
fluxes_d = np.zeros((fluxes[bands[0]].size, len(bands)))
fluxerrs_d = np.zeros((flux_errors[bands[0]].size, len(bands)))

for i, band in enumerate(bands):
    fluxes_d[:, i] = fluxes[band]
    fluxerrs_d[:, i] = flux_errors[band]

nTrain = fluxes_d.shape[0]

# Now, instead of training the SOM, we load the SOM we trained:
som_weights = np.load(f'{output_path}/{som_wide}', allow_pickle=True)
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
som = ns.NoiseSOM(metric, None, None,
                  learning=hh,
                  shape=(som_dim, som_dim),
                  wrap=False, logF=True,
                  initialize=som_weights,
                  minError=0.02)

nsubsets = 8

inds = np.array_split(np.arange(len(fluxes_d)), nsubsets)


# This function checks whether you have already run that subset, and if not it runs the SOM classifier
def assign_som(ind):
    print(f'Running rank {rank}, index {ind}')
    filename = f'{path_out}/som_wide_{som_dim}_{som_dim}_1e7_{run_name}_assign_{shear}_{rank}_subsample_{ind}.npz'
    if not os.path.exists(filename):
        print(f'Running to make {filename}')
        cells_test, _ = som.classify(fluxes_d[inds[ind]], fluxerrs_d[inds[ind]])
        np.savez(filename, cells=cells_test)
    else:
        print(f'File already exists: {filename}')


for index in range(nsubsets):
    assign_som(index)
