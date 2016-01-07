# zztop

Redshift fitter for DESI

For now, only the emission line fitter is implemented.
Features :
* redshift scan with multiple predefined lines fit
* continuum removal
* handles blended lines (OII doublet)
* simultaneous fit of line and residual continuum
* possibility to add priors or fix line ratios in fit
* fit several velocity dispersion
* several minima tracked and refitted
* select among several solutions with small delta_chi2 based on line ratio allowed ranges 
* makes use of DESI resolution matrix
* with multiprocessing

Output :
* standard DESI redshift fit format
* line fluxes and errors, chi2, for all the tracked solutions

To do :
* actual fit of the velocity dispersion for the best solution
* configuration file instead of long list of parameters in zzfit
* add other types of objects

One example to test this :

```
quickbrick -b elg --objtype=elg -n 100
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --outfile zzbest.fits --ncpu 4
plot_zzbest.py -i zzbest.fits
```

For debugging :
```
export DESI_LOGLEVEL=DEBUG

zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --use-truth --outfile zzbest-truth.fits --ncpu 4
```




