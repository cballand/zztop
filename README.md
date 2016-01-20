# zztop

Redshift fitter for DESI

For now, only the emission line fitter is implemented.

Features :
* redshift scan with multiple predefined lines fit
* continuum removal
* handles blended lines (OII doublet)
* simultaneous fit of line and residual continuum
* possibility to add priors or fix line ratios in fit
* fit several velocity dispersion in scan
* several minima tracked and refitted (inc. velocity dispersion)
* select among several solutions with small delta_chi2 based on line ratio allowed ranges 
* makes use of DESI resolution matrix
* with multiprocessing
* json configuration file with all params, see [default config file](data/zztop.json)
* prior based pca analysis of line ratios

Output :
* standard DESI redshift fit format
* line fluxes and errors, chi2, for all the tracked solutions

To do :
* add other types of objects
* tune ranking, ZWARN

One example to test this :

```
quickbrick -b elg --objtype=elg -n 100
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --outfile zzbest.fits --ncpu 4
plot_zzbest.py -i zzbest.fits
```

For debugging, examples :
```
export DESI_LOGLEVEL=DEBUG

zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --use-truth --outfile zzbest-truth.fits --ncpu 1 --first 4 --nspec 1 --conf test.json 
```

Training for line ratios :
```
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --use-truth --outfile zzbest-truth.fits --ncpu 4
zz_line_pca.py -i zzbest-truth.fits  -o pca.json
```

Then edit configuration file to use the content of the pca analysis in pca.json.
Example : [zztop_pca_prior.json](data/zztop_pca_prior.json)





