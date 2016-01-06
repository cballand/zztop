# zztop

Redshift fitter for DESI

For now, only the emission line fitter is implemented.
Features :
* redshift scan with multiple predefined lines fit
* handles blended lines (OII doublet)
* several minima tracked and refitted
* possibility to add priors or fix line ratios
* select among several solutions based on line ratio allowed ranges 
* makes use of DESI resolution matrix
* continuum removal
* fit several velocity dispersion

To do :
* configuration file instead of long list of parameters in zzfit
* add other types of objects

One example to test this :

```
quickbrick -b elg --objtype=elg -n 100
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --outfile zzbest.fits
plot_zzbest.py -i zzbest.fits
```

For debugging :
```
export DESI_LOGLEVEL=DEBUG

zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --use-truth --outfile zzbest.fits 
```




