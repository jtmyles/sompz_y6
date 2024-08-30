import pdb

import os
import glob
import numpy as np
import fitsio
import healsparse as hsp

debug = False
clobber = False
if debug:
    limit = 3
else:
    limit = None
    
datadir_chto = '/pscratch/sd/c/chto100/Cardinalv3/LSSTY1_v4_new_v2/'
outdir_jtmyles = '/pscratch/sd/j/jmyles/sompz_buzzard/cardinal/cardinal_select/'
outfile_final = os.path.join(outdir_jtmyles,'LSSTY1_WL_select.fits')

if not os.path.exists(outdir_jtmyles):
    os.system(f'mkdir -p {outdir_jtmyles}')
    
matchstr = os.path.join(datadir_chto, 'Chinchilla-3Y3a_v2.0_obs.[0-9]*.fits')
infiles = sorted( set(glob.glob(matchstr)) - set(glob.glob(matchstr.replace('fits','dnf.fits'))) )[:limit]

def selection_wl_cardinal(mag_i, mag_r, mag_r_limit, size, psf_r=0.9, imag_max=25.1):
    select_mag_i = mag_i < imag_max
    #mag_r_limit = depth_map_r # TODO
    select_mag_r = mag_r < -2.5 * np.log10(0.5) + mag_r_limit
    select_psf_r = np.sqrt(size**2  + (0.13 * psf_r)**2) > 0.1625 * psf_r

    select = select_mag_i & select_mag_r & select_psf_r
    return select
    
# load r-band depth map
infile_depth_map = '/global/cfs/cdirs/des/chto/Cardinal/LSSTdepth/lsst_y1a1_r_sp_512_depth_combined.hs'
hsp_map = hsp.HealSparseMap.read(infile_depth_map)
ninfiles = len(infiles)
list_subcats = []
ngal_sel_tot = 0
# iterate over cardinal tiles
for i, infile in enumerate(infiles):
    print(f'{i:,} of {ninfiles:,}')
    outfile_select = infile.replace(datadir_chto, outdir_jtmyles).replace('.fits','_WL_select.fits')
    outfile_subcat = infile.replace(datadir_chto, outdir_jtmyles).replace('.fits','_incl_WL_select.fits')
    if os.path.exists(outfile_select) and os.path.exists(outfile_subcat) and not clobber:
        print('Already done.')
        tmp_ = fitsio.read(outfile_subcat, columns=['ID'])
        n_ = len(tmp_)
        ngal_sel_tot += n_
        continue
    #'''
    catalog = fitsio.read(infile)

    mag_i = catalog['MAG_I']
    mag_r = catalog['MAG_R']
    size = catalog['SIZE']
    #'''

    hsp_cat = hsp_map.get_values_pos(catalog['RA'], catalog['DEC'], lonlat=True)
    
    #vpix, ra, dec = hsp_map.valid_pixels_pos(return_pixels=True)
    #pdb.set_trace()
    #hsp_map[vpix]['limmag']
    
    # determine limiting magnitude for pixel associated with each galaxy from depth map
    mag_r_limit = hsp_cat['limmag']

    select = selection_wl_cardinal(mag_i, mag_r, mag_r_limit, size, psf_r=0.9, imag_max=25.1)
    print(f'Selecting {np.sum(select):,} galaxies out of {len(mag_i):,}')
    #pdb.set_trace()
    
    # write selection

    if os.path.exists(outfile_select) and not clobber:
        print('Warning. File already exists. Nothing to write.')
        #print(f'Loading existing {outfile_select}')
        #select = fitsio.read(outfile_select)
    else:
        #select = np.array(select, dtype=[('select','f8')])
        fitsio.write(outfile_select, np.core.records.fromarrays([select], names='select'), clobber=clobber)
        print(f'Wrote {outfile_select}')
    
    # write catalog[select]
    if os.path.exists(outfile_subcat) and not clobber:
        #print('Warning. Adding HDU')
        print(f'Loading existing {outfile_select}')
        subcat = fitsio.read(outfile_subcat)
    else:
        fitsio.write(outfile_subcat, catalog[select], clobber=clobber)
        print(f'Wrote {outfile_subcat}')
        subcat = catalog[select]

print(f'{ngal_sel_tot} galaxies selected in total')
exit
assert False
for i, infile in enumerate(infiles):
    print(f'{i:,} of {ninfiles:,}')
    outfile_subcat = infile.replace(datadir_chto, outdir_jtmyles).replace('.fits','_incl_WL_select.fits')
    # append subcat to list
    subcat = fitsio.read(outfile_subcat)
    list_subcats.append(subcat)

print(f'Concatenating {len(list_subcats)} arrays')
outdata = np.concatenate(list_subcats)
if os.path.exists(outfile_final) and not clobber:
    print('Warning. Adding HDU')
    
fitsio.write(outfile_final, outdata, clobber=clobber)
print(f'Wrote {outfile_final}')    
