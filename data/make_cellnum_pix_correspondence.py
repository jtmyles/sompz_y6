import os
import numpy as np
import pandas as pd
import fitsio
from make_buzzard_deep_fields import original_pixel #, is_in_tile
print('''Warning. This script is meant to 
create a consistent hashtable relating Cardinal tile filename
identifying indices with a set of healpix pixels. 
The purposes of this script is to enable, for a given input 
(RA,DEC), to figure out which files to load to get those galaxies.
The variables are named to suggest the healpix pixels used in 
this hashtable correspond to _unrotated coordinates_. Unfortunately,
no consistent standard has been identified. So the variable names
are misnomers. The resulting hashtable is useful, but the indices 
cannot be interpreted with respect to any given coordinates.''')

import glob
#tiles = glob.glob('/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/addgalspostprocess/truth/*')
matchstr1 = '/pscratch/sd/c/chto100/DESI2/truth_v4/Chinchilla-3.[0-9]*.fits'
matchstr2 = '/pscratch/sd/c/chto100/DESI2/truth_v4_surveymags/Chinchilla-3-auxv3.*.fits'
matchstr3 = '/pscratch/sd/c/chto100/Cardinalv3/LSSTY1_v4_new_v2/Chinchilla-3Y3a_v2.0_obs.[0-9]*.fits'

NSIDE = 8
tiles = sorted( set(glob.glob(matchstr3)) - set(glob.glob(matchstr3.replace('fits','dnf.fits'))) )
#print(tiles[0])
tilenums = np.array([int(tile.split('.')[-2]) for tile in tiles])
print(tilenums)

#datadir = '/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/addgalspostprocess'
#datadir = '/pscratch/sd/c/chto100/DESI2/'
datadir = '/pscratch/sd/c/chto100/Cardinalv3/LSSTY1_v4_new_v2'

original_healpix_pixels = np.zeros(len(tilenums))
#original_healpix_pixels = np.zeros((len(tilenums), 5))
for i, tilenum in enumerate(tilenums):
    print('begin', i, tilenum)
    true_pos = os.path.join(datadir, f'Chinchilla-3Y3a_v2.0_obs.{tilenum}.fits')
    data_true_pos, header_true_pos = fitsio.read(true_pos, columns = ['RA', 'DEC', 'ID', 'Z', 'SIZE'], header=True)
    
    # this loop below is unnecessary, it's included to illustrate that different positions on a rotated tile in general are drawn from multiple unrotated tiles
    # effectively, by overwriting original_healpix_pixels[i], the result is that we take the 5th of 5 random positions and see what unrotated tile that
    # was derived from and assign that unrotated tile as being the pair of the given rotated tile
    # the 'Minimal_example_redshift_uncertainty.ipynb' notebook keeps searching until it finds a position on a rotated tile for which the drawn rectangle
    # comes from a single unrotated tile.

    rd_ids = np.random.choice(len(data_true_pos), 5)
    for j,rd_id in enumerate(rd_ids):
        rd_id = np.random.choice(len(data_true_pos), 1)

        ra_deg = data_true_pos[rd_id]['RA'][0]
        dec_deg = data_true_pos[rd_id]['DEC'][0]

        #original_healpix_pixels[i] = original_pixel(NSIDE, ra_deg, dec_deg, cardinal=True)
        original_healpix_pixels[i] = original_pixel(NSIDE, ra_deg, dec_deg, cardinal=True)
        #print(tilenum, j, original_healpix_pixels[i])
        print(tilenum, j, original_healpix_pixels[i])
    
    
print(original_healpix_pixels)

new_corresp = pd.DataFrame({'cellnum' : tilenums, 'pixel' : original_healpix_pixels.astype(int)}).set_index('cellnum')
new_corresp.to_hdf('cellnum_pix_correspondence_2.0_nest_2024-06-06.h5', 'df')
