import pdb
import os
import datetime
import h5py
import glob

import fitsio

import numpy as np

clobber = False
debug = False

chunksize = 1_000_000 #if not debug else 100000
use_mpi = False
datadir = '/pscratch/sd/j/jmyles/sompz_buzzard/cardinal/cardinal_select/'
infiles = np.array(sorted(glob.glob(os.path.join(datadir, 'Chinchilla-3Y3a_v2.0_obs.*_incl_WL_select.fits'))))
indices = np.array([int(ele.split('_')[-4].split('.')[-1]) for ele in infiles])
order = np.argsort(indices)
infiles = infiles[order]
suffix = '2024-06-25'

outfile = os.path.join(datadir, f'Chinchilla-3Y3a_v2.0_obs.comb_incl_WL_select_{suffix}.h5')
outfile_subsample = outfile.replace('.h5','_subsample.h5')
outfile_subsample_tiny = outfile.replace('.h5','_subsample_tiny.h5')

makecat = True and (not os.path.exists(outfile) or clobber)
makesubcat = True and (not os.path.exists(outfile_subsample) or clobber)
makesubcat_tiny = True and (not os.path.exists(outfile_subsample_tiny) or clobber)

makebools = [makecat, makesubcat, makesubcat_tiny]
make_outfiles = [outfile, outfile_subsample, outfile_subsample_tiny]

for makebool, make_outfile in zip(makebools, make_outfiles):
    if makebool:
        print(f'Making   -- {make_outfile}')
    else:
        print(f'Skipping -- {make_outfile}')

if debug:
    outfile = outfile.replace('.h5','_debug.h5')
    outfile_subsample = outfile_subsample.replace('.h5','_debug.h5')
    outfile_subsample_tiny = outfile_subsample_tiny.replace('.h5','_debug.h5')
    infiles = infiles[:5]

ninfiles = len(infiles)    
#f = h5py.File(wide_field_file,'r') # this is the master catalog
total_length = 898_490_228 #if not debug else 100 * chunksize
total_length_subsample = 10_000_000
total_length_subsample_tiny = 1_000_000

columns_phot = [f'{col_}_{band}' for band in 'UGRIZY' for col_ in ['MAG','MAGERR','FLUX','IVAR']]
columns_read = columns_phot + ['ID', 'RA', 'DEC', 'SIZE', 'Z', ]
columns_write = columns_read + [f'{col_}_{band}' for band in 'UGRIZY' for col_ in ['FLUX_ERR']]
dtypes = [np.float32,] * len(columns_phot) + [int,] + [np.float32, ] * (len(columns_write) - len(columns_phot) - 1)

# MPI setup
if use_mpi == True:
    import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank=0
    size=1

print("rank, size=",rank,size)
parent = '' # 'catalog/'
if rank==0:
    # open outfiles
    if makecat:
        fout = h5py.File(outfile,'w')
    if makesubcat:
        fout_subsample = h5py.File(outfile_subsample,'w')
    if makesubcat_tiny:
        fout_subsample_tiny = h5py.File(outfile_subsample_tiny,'w')
    # make datasets
    for col, dtype in zip(columns_write, dtypes):
        #print(f'Make column {parent}{col}')
        if makecat:
            fout.create_dataset(f'{parent}{col}',
                                maxshape=(total_length,),
                                shape=(total_length,), dtype=dtype, chunks=(chunksize,) )
        if makesubcat:
            fout_subsample.create_dataset(f'{parent}{col}',
                                          maxshape=(total_length_subsample,),
                                          shape=(total_length_subsample,),
                                          dtype=dtype, chunks=(chunksize,) )
        if makesubcat_tiny:
            fout_subsample_tiny.create_dataset(f'{parent}{col}',
                                               maxshape=(total_length_subsample_tiny,),
                                               shape=(total_length_subsample_tiny,),
                                               dtype=dtype, chunks=(10_000,) )

if use_mpi == True:
    comm.Barrier()

# indices in catalog that correspond to breaks between subarrays
# associated with each MPI process
breaks = np.linspace(0,total_length-1,num=size,endpoint=True).astype(int)
if len(breaks)==1:
    breaks = [0,total_length]
print(breaks)

# end index for last subarray
slice_end = total_length if rank == size - 1 else breaks[rank + 1]

print(f'rank {rank} . size {size} . breaks {breaks} . slice_end {slice_end}')

tmp = np.random.choice(total_length, total_length_subsample, replace=False)
idx_subsample = np.array(sorted(tmp))
idx_subsample_tiny = np.array(sorted(tmp[:total_length_subsample_tiny]))

# load fits files and place data into new h5 file
starttime = datetime.datetime.now()
print(f'Start {starttime}')
idx_start = 0
idx_start_subsample = 0
idx_start_subsample_tiny = 0
for i, infile in enumerate(infiles):
    print(f'{datetime.datetime.now() - starttime}: File {i:03d} of {ninfiles:03d}: {infile}')
    data_ = fitsio.read(infile, columns=columns_read)
    #pdb.set_trace()
    # determine indices to write into
    nthis = len(data_)
    idx_end = idx_start+nthis
    # determine indices to read from for subsample
    #idx_subsample_this = np.where((idx_subsample >= idx_start) & (idx_subsample < idx_end))[0] - idx_start
    idx_subsample_this = idx_subsample[(idx_subsample >= idx_start) & (idx_subsample < idx_end)] - idx_start
    idx_subsample_tiny_this = idx_subsample_tiny[(idx_subsample_tiny >= idx_start) & (idx_subsample_tiny < idx_end)] - idx_start
    nthis_subsample = len(idx_subsample_this)
    nthis_subsample_tiny = len(idx_subsample_tiny_this)
    # determine indices to write into for subsample
    idx_end_subsample = idx_start_subsample+nthis_subsample
    idx_end_subsample_tiny = idx_start_subsample_tiny+nthis_subsample_tiny
    
    print(f'idx start {idx_start}. idx end {idx_end}')
    print(f'idx start sub {idx_start_subsample}. idx end sub {idx_end_subsample}')
    print(f'idx start sub tiny {idx_start_subsample_tiny}. idx end sub tiny {idx_end_subsample_tiny}')
    for col in columns_write:
        if col[:len('FLUX_ERR')] == 'FLUX_ERR':
            print(f'making {col} from IVAR')
            if makecat: fout[f'{parent}{col}'][idx_start:idx_end] = 1./np.sqrt(data_[col.replace('FLUX_ERR','IVAR')])
            if makesubcat:
                fout_subsample[f'{parent}{col}'][idx_start_subsample:idx_end_subsample] = 1./np.sqrt(data_[idx_subsample_this][col.replace('FLUX_ERR','IVAR')])
            if makesubcat_tiny:
                fout_subsample_tiny[f'{parent}{col}'][idx_start_subsample_tiny:idx_end_subsample_tiny] = 1./np.sqrt(data_[idx_subsample_tiny_this][col.replace('FLUX_ERR','IVAR')])
        else:
            print(f'copying {col}')
            if makecat: fout[f'{parent}{col}'][idx_start:idx_end] = data_[col]
            if makesubcat: fout_subsample[f'{parent}{col}'][idx_start_subsample:idx_end_subsample] = data_[idx_subsample_this][col]
            if makesubcat_tiny: fout_subsample_tiny[f'{parent}{col}'][idx_start_subsample_tiny:idx_end_subsample_tiny] = data_[idx_subsample_tiny_this][col]
            
    # update start indices
    idx_start += nthis
    idx_start_subsample += nthis_subsample
    idx_start_subsample_tiny += nthis_subsample_tiny    
if makecat:
    fout.close()
    print(f'Wrote {outfile}')
if makesubcat:
    fout_subsample.close()
    print(f'Wrote {outfile_subsample}')
if makesubcat_tiny:
    fout_subsample_tiny.close()
    print(f'Wrote {outfile_subsample_tiny}')

