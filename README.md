# zztop
redhisft fitter for massive spectroscopic surveys

For now, only the emission line fitter is implemented.

One example to test this :

```
quickbrick -b elg --objtype=elg -n 100
zzfit --b brick-b-elg.fits --r brick-r-elg.fits --z brick-z-elg.fits --outfile zzbest.fits
plot_zzbest.py -i zzbest.fits
```




