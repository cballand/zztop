# zztop

Redshift fitter for DESI

For now, only the emission line fitter is implemented.
Features :
* redshift scan with multiple predefined lines fit
* handles blended lines (OII doublet)
* several minima tracked and refitted
* possibility to add priors to line ratios 
* makes use of DESI resolution matrix

One example to test this :

```
quickbrick -b elg --objtype=elg -n 100
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --outfile zzbest.fits
plot_zzbest.py -i zzbest.fits
```




