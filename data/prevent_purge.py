import glob
#import fitsio
import astropy.io.fits as fits

matchstr1 = '/pscratch/sd/c/chto100/DESI2/truth_v4/Chinchilla-3.[0-9]*.fits'
matchstr2 = '/pscratch/sd/c/chto100/DESI2/truth_v4_surveymags/Chinchilla-3-auxv3.*.fits'
matchstr3 = '/pscratch/sd/c/chto100/Cardinalv3/LSSTY1_v4_new_v2/Chinchilla-3Y3a_v2.0_obs.[0-9]*.fits'
matchstrs = [matchstr1, matchstr2, matchstr3]

infiles1 = sorted( set(glob.glob(matchstr1)) - set(glob.glob(matchstr1.replace('fits','lens.fits'))) )
infiles2 = sorted( set(glob.glob(matchstr2)) - set(glob.glob(matchstr2.replace('fits','lens.fits'))) )
infiles3 = sorted( set(glob.glob(matchstr3)) - set(glob.glob(matchstr3.replace('fits','dnf.fits'))) )

#outfile = './prevent_purge.out'
#foutfile = open(outfile, 'a')

for i, infiles in enumerate([infiles1, infiles2, infiles3]):
    print(f'There are {len(infiles)} files with names matching {matchstrs[i]}. (excluding lens.fits and dnf.fits)')

    for j, infile in enumerate(infiles):
        #print(infile)
        #data = fitsio.read(infile)
        hdulist = fits.open(infile)
        #print(hdulist)
        if j == 0:
            nhdu = len(hdulist)

            assert len(hdulist) == 2, 'There is an unexpected number of HDUs'

            data = hdulist[1].data
            nrow = len(data)
            ncol = len(data.dtype.names)
            print(f'The first such file has {nhdu} HDUs. The 1st HDU has {nrow:,} rows and {ncol} columns')
            print(f'Columns: {data.dtype.names}')
        elif j % 50 == 0:
            print(f'i: {i}. j: {j}')
    print('\n\n\n')

#foutfile.close()
#print(f'Wrote {outfile}')

infiles1 = sorted(glob.glob(matchstr1.replace('fits','lens.fits')))
infiles2 = [] # sorted(glob.glob(matchstr2))
infiles3 = sorted(glob.glob(matchstr3.replace('fits','dnf.fits')))

for infiles in [infiles1, infiles2, infiles3]:
    for j, infile in enumerate(infiles):
        if j % 50 == 0:
            print(j)
        fits.open(infile)
