import pdb
import numpy as np
import pandas as pd
import fitsio
import os, sys
import healpy as hp
import yaml
import glob
import time
from shutil import copyfile
from numpy.lib.recfunctions import append_fields
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import healpix_util as hu
import skymapper as skm
import pandas as pd
import pickle
append = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))) + '/sompz'
print(append)
sys.path.append(append)
from utils_buzzard import get_df_true, get_balrog_sample

def main():
    t0 = time.time()
    if(len(sys.argv)==1):
            cfgfile = 'make_300.cfg'
    else:
            cfgfile = sys.argv[1]

    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    indir = cfg['indir']
    outdir = cfg['outdir']

    square_size_list = cfg['square_size_list']
    redshift_sample_bool = cfg['redshift_sample_bool']

    if ('min_gal_redshift_sample' and 'max_gal_redshift_sample') in list(cfg.keys()):
        min_gal_redshift_sample = cfg['min_gal_redshift_sample']
        max_gal_redshift_sample = cfg['max_gal_redshift_sample']
        bool_fixed_n_gal = False
    elif 'fixed_n_gal_redshift_sample' in list(cfg.keys()):
        min_gal_redshift_sample = cfg['fixed_n_gal_redshift_sample']
        max_gal_redshift_sample = np.inf
        bool_fixed_n_gal = True
    else:
        raise ValueError('Either a fixed number of galaxies or a window must be specified for the redshift sample')

    GalBaseName = cfg['GalBaseName']
    MagBaseName = cfg['MagBaseName']
    # zero point of magnitudes
    zp = cfg['zp']
    corresp_file = cfg['buzzard_CellNumHealPixCorrespondence']
    print('Size deep field(s): ', square_size_list)
    print('Size redshift field(s): ', np.array(square_size_list)[np.array(redshift_sample_bool)])

    ndeepfields = cfg['buzzard_ndeep_realizations']
    os.system('mkdir -p {}'.format(outdir))
    for job_id in range(ndeepfields):
        # Path to store true fluxes of the deep field galaxies
        new_despz_path = '{0}/{1}.{2}.fits'.format(outdir, MagBaseName, job_id)
        # Path to store true positions and redshift of the deep field galaxies
        new_true_path = '{0}/{1}.{2}.fits'.format(outdir, GalBaseName, job_id)

        if os.path.exists(new_despz_path) and os.path.exists(new_true_path):
            print(new_despz_path)
            print(new_true_path)
            print(str(job_id) + ' already exists')
            continue

        # select deep fields. resulting tables are concatenated for all fields on the sky
        data_pos, header_pos, data_mags, header_mags = select_deep_fields_large(square_size_list, redshift_sample_bool,
                                                                                indir, outdir, t0,
                                                                                imag_max=27, # LSST Y10 depth
                                                                                copy=False,
                                                                                corresp_file=corresp_file)
        print(job_id, time.time()-t0, 'Deep fields obtained. Now start printing and plotting diagnostics')
        print('Number of galaxies in the deep fields: ', data_pos.shape)

        tiles = np.unique(data_pos['TILE'])
        #hpix = hu.HealPix("ring", 4096)
        NSIDE = 4096
        pixarea = hp.nside2pixarea(NSIDE, degrees=False, ) # square radian
        for cell in tiles: # in cellnums
            df = data_pos[data_pos['TILE']==cell]
            theta_ = np.radians(90 - df['DEC'])
            phi_ = np.radians(df['RA'])
            #pixnums = hpix.eq2pix(df['RA'], df['DEC'])
            pixnums = hp.ang2pix(NSIDE, theta_, phi_)
            print('size = {:.3f}'.format(len(set(pixnums))*pixarea*(180/np.pi)**2)) # hpix.area
            print('n gals = {}'.format(len(df)))
        else:
            print(f'finished going through all cells in tiles {tiles} making up the deep fields')
        assert len(df) > 0
        # Let's see our deep fields
        #pixnums = hpix.eq2pix(data_pos['RA'], data_pos['DEC'])
        theta_ = np.radians(90 - data_pos['DEC'])
        phi_ = np.radians(data_pos['RA'])
        #pixnums = hp.ang2pix(NSIDE, -data_pos['DEC']+90, data_pos['RA'])
        pixnums = hp.ang2pix(NSIDE, theta_, phi_)
        map_df = np.zeros(hp.nside2npix(NSIDE))
        map_df[pixnums] = 1

        #fig = plt.figure(figsize=(12,8))
        #ax = fig.add_subplot(111, aspect='equal')
        #fig, ax, proj = skm.plotHealpix(map_df, NSIDE, ax=ax)
        crit = skm.stdDistortion
        proj = skm.Albers.optimize(df['RA'], df['DEC'], crit=crit)
        skm_map = skm.Map(proj)
        #fig, ax, proj = skm_map.healpix(map_df, NSIDE, ax=ax)
        out = skm_map.healpix(map_df)
        fig = out.get_figure()
        fig.savefig(outdir + '/mock_deep_fields.{}.png'.format(job_id), dpi=300)
        plt.close()

        if not os.path.exists(outdir):
            print('Create directory: ', outdir)
            os.makedirs(outdir)

        # Write deep fields to disk as individual position/magnitude FITS files

        fits_despz = fitsio.FITS(new_despz_path,'rw')
        fits_despz.write(data_mags, header=header_mags)
        fits_despz.close()
        print(time.time()-t0, 'Deep fields fluxes saved :', new_despz_path)

        fits_true = fitsio.FITS(new_true_path,'rw')
        fits_true.write(data_pos, header=header_pos)
        fits_true.close()
        print(time.time()-t0, 'Deep fields positions/redshifts saved :', new_true_path)

        # Write single deep field catalog
        df_true_pos, df_true_mag = get_df_true(new_despz_path, new_true_path, cfg['bands_name_file'])
        df_true = pd.concat([df_true_pos, df_true_mag], axis=1)
        df_true.to_pickle(cfg['deep_file'].format(job_id))

        
def get_random_square_in_buzzard(square_size):
    if(square_size>5000.):
        print("cannot cut square of size",square_size,"from Buzzard")
        sys.exit(1)
    print('Attempting getting random square in Buzzard')
    phi = np.random.uniform(0,np.pi) # R.A. from 0 to 180 deg
    costheta = np.random.uniform(-1,1) # Dec. from 180 to 0 deg
    theta = np.arccos(costheta)
    
    ramin,decmin = rad_to_deg(phi),rad_to_deg(theta)
    decmax = decmin + np.sqrt(square_size)
    # ? if (decmax>0):
    if(decmax>90.): # impose cut on dec at 90 deg
        return get_random_square_in_buzzard(square_size)

    # CHANGE COORDINATES. move square to Cardinal nominal coords
    print('Moving square from Buzzard sky to Cardinal Sky')
    decmin = -decmin
    decmax = decmin + np.sqrt(square_size)
    if (ramin > 90) & (ramin < 270): # Buzzard unrotated spanned from [0,180]. Cardinal spans [0,90] U [270,360]
        ramin = ramin + 180
    
    # DeltaRA * int_decmin^decmax cos(dec) ddec == square_size
    # DeltaRA = square_size/(sin(decmax)-sin(decmin))
    DeltaRA = square_size/(rad_to_deg(np.sin(deg_to_rad(decmax))-np.sin(deg_to_rad(decmin))))
    ramax = ramin+DeltaRA

    if(ramax>180.):
        return get_random_square_in_buzzard(square_size)
    
    return ramin,decmin,ramax,decmax
    
def get_cellnums_from_square(ramin,decmin,ramax,decmax, cardinal=False, corresp_file=None):
    ra,dec = np.meshgrid(np.arange(ramin,ramax,0.1),np.arange(decmin,decmax,0.1))
    # convert to index of NSIDE=8 nested healpix map
    idx = hp.ang2pix(8,np.radians(-dec+90.),np.radians(ra),nest=True)
    idx = np.unique(idx)

    print(f'unique tiles in square {idx}')
    
    if cardinal:
        cellnums = []
        corresp = pd.read_hdf(corresp_file, mode='a')
        print('found tiles in corresp file', np.intersect1d(idx, corresp['pixel'].values))
        for cellnum_ in corresp.index:
            if len(np.intersect1d(idx, corresp.loc[cellnum_]) > 0):
                cellnums.append(cellnum_)
        print('filename indices to use for these tiles', cellnums)
        return cellnums
    else:
        return idx

def read_deep_tile(cellnum, P1_ra_deg, P1_dec_deg, P2_ra_deg, P2_dec_deg, true_pos, true_mag, t0, imag_max=24.5):
    # read Buzzard tile 
    print(time.time()-t0, 'Reading tile {0}'.format(cellnum))
    columns_add_cardinal = [f'{col_}_{band}' for band in 'UGRIZY' for col_ in ['MAG','MAGERR','FLUX','IVAR']] # cfg['Bands']
    data_true_pos, header_true_pos = fitsio.read(true_pos, columns=['RA', 'DEC', 'ID', 'Z', 'SIZE'] + columns_add_cardinal, 
                                                 header=True) # include INDEX for older versions of buzzard
    #print(time.time()-t0, 'Positions read')
    radec_conds = ((data_true_pos['RA']>P1_ra_deg) & (data_true_pos['RA']<P2_ra_deg) & (data_true_pos['DEC']>P1_dec_deg) & (data_true_pos['DEC']<P2_dec_deg))
    print(time.time()-t0, f'Radec conditions computed. selecting {np.sum(radec_conds)} of {len(radec_conds)} galaxies in this tile/file based on ra/dec conds')
    data_true_pos = data_true_pos[radec_conds]
    
    #read the true magnitude file (load only galaxies that are in the square)
    data_true_mags, header_true_mags = fitsio.read(true_mag, rows=np.where(radec_conds)[0], columns=['LMAG',], header=True) # 'FLUX'
    #print(time.time()-t0, 'Data true mags file of tile {0} loaded'.format(cellnum))
    
    # selection criteria for our "deep field" catalog
    # cut both catalogs
    #conds = ((data_true_pos['Z'] < 1.5) & (data_true_mags['LMAG'][:,3]<imag_max) & np.isfinite(data_true_mags['LMAG'][:,3]))
    idx_i = 2 # was 3 with buzzard. is 2 with cardinal
    conds = ((data_true_mags['LMAG'][:,idx_i]<imag_max) & np.isfinite(data_true_mags['LMAG'][:,idx_i]))

    '''
    TODO
    infile_depth_map = '/global/cfs/cdirs/des/chto/Cardinal/LSSTdepth/lsst_y1a1_r_sp_512_depth_combined.hs'
    hsp_map = hsp.HealSparseMap.read(infile_depth_map)
    mag_i = catalog['MAG_I']
    mag_r = catalog['MAG_R']
    size = catalog['SIZE']

    hsp_cat = hsp_map.get_values_pos(catalog['RA'], catalog['DEC'], lonlat=True)
    
    # determine limiting magnitude for pixel associated with each galaxy from depth map
    mag_r_limit = hsp_cat['limmag']

    select = selection_wl_cardinal(mag_i, mag_r, mag_r_limit, size, psf_r=0.9, imag_max=25.1)
    print(f'Selecting {np.sum(select):,} galaxies out of {len(mag_i):,}')

    conds &= select
    '''
    #conds = np.isfinite(data_true_mags['LMAG'][:,3])
    #print(time.time()-t0, 'Mag/Z conditions computed')
    data_true_pos = data_true_pos[conds]
    #print(time.time()-t0, 'Pos file cut on Mag/Z conditions')
    data_true_mags = data_true_mags[conds]
    #print(time.time()-t0, 'Mag file cut on Mag/Z conditions')
    print(time.time()-t0, f'{len(data_true_mags)} galaxies from this healpix pixel i.e. cardinal tile remain after mag cuts')
    return data_true_pos, header_true_pos, data_true_mags, header_true_mags

def select_deep_fields_large(square_size_list, redshift_sample_bool, indir, outdir, t0, copy=True, imag_max=24.5, corresp_file=None):
    """Cut the deep fields, not limiting them to a single Buzzard tile
    
    Parameters
    ----------
    square_size_list :        List of float. The sizes in square degrees of the deep fields.
    redshift_sample_bool :    List of boolean. True if the field is also used as redshift sample.
    bool_fixed_n_gal :        Boolean. True if the number of galaxies in the redshift sample is fixed.
    min_gal_redshift_sample : Int. The minimum number of galaxies in the redshift sample. If bool_fixed_n_gal=True, the number of galaxies is fixed to min_gal_redshift_sample.
    max_gal_redshift_sample : Int. The maximum number of galaxies in the redshift sample. If bool_fixed_n_gal=True, this is unused.
    indir :                   String. Path where the Buzzard catalog is located.
    outdir :                  String. Path where the Buzzard catalog are copied (probably on scratch). Unused if copy=False.
    t0 :                      Float. Time.
    copy :                    Boolean. Specify if the Buzzard files must be copied to outdir.
    """
    
    # we will work in unrotated coordinates here, i.e. Buzzard spans RA,dec = [0,180],[0,90]
    
    d_pos, h_pos, d_mags, h_mags = [], [], [], []

    for square_size, redshift_bool in zip(square_size_list, redshift_sample_bool): # for each of the deep fields
        ramin,decmin,ramax,decmax = get_random_square_in_buzzard(square_size)
        print(f'random square -- ramin {ramin:.2f} -- decmin {decmin:.2f} -- ramax {ramax:.2f} -- decmax {decmax:.2f}')
        cellnums = get_cellnums_from_square(ramin,decmin,ramax,decmax, cardinal=True, corresp_file=corresp_file)
        print('cellnums', cellnums)
        print(time.time()-t0, 'Tile list: {0}. Begin loop over these tiles.'.format(cellnums))
        
        for cellnum in cellnums:
            # TODO: read in griz from true_pos. read in all other mags from true_mag
            # original location of the Buzzard tile
            #true_pos = indir + '/truth/Chinchilla-3_lensed_rs_scat_cam.{0}.fits'.format(cellnum) # take positions / Z from here
            #true_mag = indir[:-18] + '/surveymags/Chinchilla-3-v1.9.7-auxmag.{0}.fits'.format(cellnum) # take magnitudes from here
            #true_pos = os.path.join(indir, '/truth_v4/Chinchilla-3.{0}.fits'.format(cellnum))
            true_mag = os.path.join(indir, f'DESI2/truth_v4_surveymags/Chinchilla-3-auxv3.{cellnum}.fits')
            true_pos = os.path.join(indir, f'Cardinalv3/LSSTY1_v4_new_v2/Chinchilla-3Y3a_v2.0_obs.{cellnum}.fits') # the truth catalogs do not have ra,dec info for the latest Cardinal run
            
            if copy:
                # copy Buzzard tile to scratch
                #cp_true_pos = outdir + '/Chinchilla-0Y3_v1.6_truth.{0}.fits'.format(cellnum)
                #cp_true_mag = outdir + '/Chinchilla-0_DESY3PZ.{0}.fits'.format(cellnum)
                cp_true_pos = os.path.join(outdir,'Chinchilla-3.{0}.fits'.format(cellnum))
                cp_true_mag = os.path.join(outdir,'Chinchilla-3_LSSTY1PZ.{0}.fits'.format(cellnum))

                print(time.time()-t0, 'Start copying files to scratch')
                copyfile(true_pos, cp_true_pos)
                copyfile(true_mag, cp_true_mag)
                print(time.time()-t0, 'Files copied to scratch')

                true_pos = cp_true_pos
                true_mag = cp_true_mag
                
            # read whatever is in that square from the full tile
            # cellnum, P1_ra_deg, P1_dec_deg, P2_ra_deg, P2_dec_deg, true_pos, true_mag,imag_max=24.5
            data_true_pos, header_true_pos, data_true_mags, header_true_mags = read_deep_tile(cellnum, ramin, decmin, ramax, decmax, true_pos, true_mag, t0, imag_max=imag_max)
            print(time.time()-t0, 'Tile read')
                
            if copy:
                # Delete Buzzard tile from scratch
                os.remove(true_pos)
                os.remove(true_mag)
                print(time.time()-t0, 'Files deleted from scratch')
                
            #add the tile number as a column of the data
            print(time.time()-t0, 'True data of deep field tile loaded. Tile : {0}.'.format(cellnum))
            data_true_pos = append_fields(data_true_pos, 'TILE', data=len(data_true_pos)*[cellnum], dtypes='>i8', usemask=False) # use big-endian int to match fits 
            data_true_pos = append_fields(data_true_pos, 'REDSHIFT FIELD', data=len(data_true_pos)*[redshift_bool], dtypes='>i8', usemask=False) # use big-endian int to match fits
            d_pos.append(data_true_pos)
            h_pos.append(header_true_pos)
            d_mags.append(data_true_mags)
            h_mags.append(header_true_mags)
        #pdb.set_trace()
    mags_data = np.concatenate(d_mags)

    # convert ndarray to pandas.DataFrame
    d_pos_df = [pd.DataFrame(ele.byteswap().newbyteorder()) for ele in d_pos]
    # and cast ID to int
    d_pos_df = [x.astype({'ID' : int}) for x in d_pos_df]
    pos_data = pd.concat(d_pos_df)
    pos_data = pos_data.to_records()
    # pdb.set_trace() # clear
    return pos_data, h_pos[0], mags_data, h_mags[0]

# def select_deep_fields(cellnums, square_size_list, redshift_sample_bool, bool_fixed_n_gal, min_gal_redshift_sample, max_gal_redshift_sample, indir, outdir, t0, copy=True):
#     """Cut the deep fields in a Buzzard tile

#         Parameters
#         ----------
#         cellnums :                List of int. The Buzzard tile number where the deep fields must be cut.
#         square_size_list :        List of float. The sizes in square degrees of the deep fields.
#         redshift_sample_bool :    List of boolean. True if the field is also used as redshift sample.
#         bool_fixed_n_gal :        Boolean. True if the number of galaxies in the redshift sample is fixed.
#         min_gal_redshift_sample : Int. The minimum number of galaxies in the redshift sample. If bool_fixed_n_gal=True, the number of galaxies is fixed to min_gal_redshift_sample.
#         max_gal_redshift_sample : Int. The maximum number of galaxies in the redshift sample. If bool_fixed_n_gal=True, this is unused.
#         indir :                   String. Path were the Buzzard catalog is located.
#         outdir :                  String. Path were the Buzzard catalog are copied (probably on scratch). Unused if copy=False.
#         t0 :                      Float. Time.
#         copy :                    Boolean. Specify if the Buzzard files must be copied to outdir.
#         """
    
#     if bool_fixed_n_gal:
#         max_gal_redshift_sample = np.inf
        
#     d_pos, h_pos, d_mags, h_mags = [], [], [], []
    
#     for cellnum, square_size, redshift_bool in zip(cellnums, square_size_list, redshift_sample_bool): # for each of the deep fields
            
#         # original location of the Buzzard tile
#         #true_pos = indir + '/truth/Chinchilla-3_lensed_rs_scat_cam.{0}.fits'.format(cellnum) # take positions / Z from here
#         #true_mag = indir[:-18] + '/surveymags/Chinchilla-3-v1.9.7-auxmag.{0}.fits'.format(cellnum) # take magnitudes from here
#         true_pos = os.path.join(indir, 'DESI2/truth_v4/Chinchilla-3.{0}.fits'.format(cellnum))
#         true_mag = os.path.join(indir, 'DESI2/truth_v4_surveymags/Chinchilla-3-auxv3.{0}.fits'.format(cellnum))
        
#         if copy:
#             # copy Buzzard tile to scratch
#             #cp_true_pos = outdir + '/Chinchilla-3_lensed_rs_scat_cam.{0}.fits'.format(cellnum)
#             #cp_true_mag = outdir + '/Chinchilla-3_DESY3PZ.{0}.fits'.format(cellnum)
#             cp_true_pos = outdir + 'Chinchilla-3.{0}.fits'.format(cellnum)
#             cp_true_mag = outdir + 'Chinchilla-3_LSSTY1PZ.{0}.fits'.format(cellnum)

#             print(time.time()-t0, 'Start copying files to scratch')
#             copyfile(true_pos, cp_true_pos)
#             copyfile(true_mag, cp_true_mag)
#             print(time.time()-t0, 'Files copied to scratch')
        
#             true_pos = cp_true_pos
#             true_mag = cp_true_mag
        
#         #cut the deep field
#         data_true_pos, header_true_pos, data_true_mags, header_true_mags = select_deep_field(cellnum, square_size, true_pos, true_mag, t0)
#         if redshift_bool: #if the deep field is also the redshift sample, make sure
#             while (len(data_true_pos) < min_gal_redshift_sample) or (len(data_true_pos) > max_gal_redshift_sample):
#                 print('Not enough/too many galaxies in the redshift sample. We want {0} < n_gal < {1}. But we have {2}'.format(min_gal_redshift_sample, max_gal_redshift_sample, len(data_true_pos)))
#                 data_true_pos, header_true_pos, data_true_mags, header_true_mags = select_deep_field(cellnum, square_size, true_pos, true_mag, t0)
               
#             if bool_fixed_n_gal:
#                 choice = np.random.choice(len(data_true_pos), min_gal_redshift_sample, replace=False)
#                 data_true_pos = data_true_pos[choice]
#                 data_true_mags = data_true_mags[choice]  
        
#         if copy:
#             # Delete Buzzard tile from scratch
#             os.remove(true_pos)
#             os.remove(true_mag)
#             print(time.time()-t0, 'Files deleted from scratch')
        
#         #add the tile number as a column of the data
#         print(time.time()-t0, 'True data of deep field loaded. Tile : {0}. Size : {1}'.format(cellnum, square_size))
#         data_true_pos = append_fields(data_true_pos, 'TILE', data=len(data_true_pos)*[cellnum], dtypes=int, usemask=False)
#         data_true_pos = append_fields(data_true_pos, 'REDSHIFT FIELD', data=len(data_true_pos)*[redshift_bool], dtypes=int, usemask=False)
#         d_pos.append(data_true_pos)
#         h_pos.append(header_true_pos)
#         d_mags.append(data_true_mags)
#         h_mags.append(header_true_mags)

#     pos_data = np.concatenate(d_pos)
#     mags_data = np.concatenate(d_mags)
    
#     return pos_data, h_pos[0], mags_data, h_mags[0]


# def select_deep_field(cellnum, square_size, true_pos, true_mag, t0, imag_max=24.5):
    
#     # read Buzzard tile 
#     data_true_pos, header_true_pos = fitsio.read(true_pos, columns=['RA', 'DEC', 'ID', 'Z', 'SIZE'], header=True)
#     print(time.time()-t0, 'Data true position file of tile {0} loaded'.format(cellnum))

#     #select galaxies in a square of 'square_size' deg^2 in Buzzard tile
#     square_in_tile = False
#     # the four corner of the square must be in the tile 
#     while not square_in_tile:
#         #pick randomly a galaxy. Use its position as the first corner of the square
#         rd_id = np.random.choice(len(data_true_pos),1)
#         P1_ra_deg = data_true_pos[rd_id]['RA'][0]
#         P1_dec_deg = data_true_pos[rd_id]['DEC'][0]
#         print(time.time()-t0, 'Selected position in radec : ', P1_ra_deg, P1_dec_deg)
        
#         #compute position in rad of the corner
#         P1_dec_rad = deg_to_rad(P1_dec_deg)
#         d = deg_to_rad(np.sqrt(square_size))

#         #compute position of opposite corner in deg
#         P2_ra_deg = P1_ra_deg + np.sqrt(square_size)/np.cos(P1_dec_rad+d/2.)
#         P2_dec_deg = P1_dec_deg + np.sqrt(square_size)

#         #Check that the 4 corners are in the tile
#         square_in_tile = is_in_tile(P1_ra_deg, P1_dec_deg, cellnum) & is_in_tile(P1_ra_deg, P2_dec_deg, cellnum) & is_in_tile(P2_ra_deg, P1_dec_deg, cellnum) & is_in_tile(P2_ra_deg, P2_dec_deg, cellnum)    
#     print(time.time()-t0, 'Selected square is in tile')
    
#     #cut the position file on ra and dec to keep only the square
#     pdb.set_trace()
#     radec_conds = ((data_true_pos['RA']>P1_ra_deg) & (data_true_pos['RA']<P2_ra_deg) & (data_true_pos['DEC']>P1_dec_deg) & (data_true_pos['DEC']<P2_dec_deg))
#     print(time.time()-t0, 'Radec conditions computed')
#     data_true_pos = data_true_pos[radec_conds]
#     print(time.time()-t0, 'Pos file cut on radec conditions')
    
#     #read the true magnitude file (load only galaxies that are in the square)
#     data_true_mags, header_true_mags = fitsio.read(true_mag, rows=np.where(radec_conds), columns=['LMAG', 'FLUX'], header=True)    
#     print(time.time()-t0, 'Data true mags file of tile {0} loaded'.format(cellnum))
    
#     # selection criteria for our "deep field" catalog
#     # cut both catalogs
#     #conds = ((data_true_pos['Z'] < 1.5) & (data_true_mags['LMAG'][:,3]<24.5) & np.isfinite(data_true_mags['LMAG'][:,3]))
#     conds = ((data_true_mags['LMAG'][:,3]<imag_max) & np.isfinite(data_true_mags['LMAG'][:,3]))
#     #conds = np.isfinite(data_true_mags['LMAG'][:,3])
    
#     print(time.time()-t0, 'Mag/Z conditions computed')
#     data_true_pos = data_true_pos[conds]
#     print(time.time()-t0, 'Pos file cut on Mag/Z conditions')
#     data_true_mags = data_true_mags[conds]
#     print(time.time()-t0, 'Mag file cut on Mag/Z conditions')

#     return data_true_pos, header_true_pos, data_true_mags, header_true_mags
 
#check if the position ra, dec is in cellnum
#def is_in_tile(ra, dec, cellnum, corresp_file):
#    corresp = pd.read_hdf(corresp_file, mode='a')
#    true_pix = corresp.loc[cellnum].values[0]
#    NSIDE = 8
#    pixel = original_pixel(NSIDE, ra, dec)
#    if true_pix==pixel:
#        return True
#    else:
#        return False

def original_pixel(NSIDE, ra, dec, cardinal=False):
    rmat = np.linalg.inv(np.array([[-0.33976859,  0.93565351,  0.0954453 ],
                     [ 0.94049749,  0.33851385,  0.02954402],
                     [-0.00466659,  0.09980419, -0.99499615]]))
    theta=np.radians(-dec+90.)
    phi=np.radians(ra)
    if cardinal:
        tr, pr = theta, phi
    else:
        tr, pr = hp.rotator.rotateDirection(rmat, theta, phi)
    pixel = hp.pixelfunc.ang2pix(NSIDE, tr, pr, nest=True) # nest=False by default, so RING by default
    return int(pixel)

def deg_to_rad(deg):
    return deg*np.pi/180

def rad_to_deg(rad):
    return rad*180/np.pi

if __name__ == '__main__':
    main()
