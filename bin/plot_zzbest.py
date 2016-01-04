#!/usr/bin/env python

from astropy.io import fits
import argparse
import pylab

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path to zzbest.fits file')
    args = parser.parse_args()
    
    hdulist=fits.open(args.infile)
    hdulist.info()
    print hdulist[1].columns.names
    
    
    bestz=hdulist[1].data["Z"]
    errz=hdulist[1].data["ZERR"]
    
    truez=hdulist["_TRUTH"].data["TRUEZ"]
    n=min(bestz.size,truez.size)
    deltaz=bestz[:n]-truez[:n]
    errz=errz[:n]
    
    pylab.figure()
    a=pylab.subplot(1,1,1)
    #a.plot(truez[:n],deltaz,"o")
    a.errorbar(truez[:n],deltaz,errz,fmt="o")
    
    a.set_xlabel("Redshift")
    a.set_ylabel("Best - True Redshift")
    
    pylab.show()


if __name__ == '__main__':
    main()
    

