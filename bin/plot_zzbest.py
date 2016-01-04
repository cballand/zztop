#!/usr/bin/env python

from astropy.io import fits
import argparse
import pylab
import numpy as np

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path to zzbest.fits file')
    args = parser.parse_args()
    
    hdulist=fits.open(args.infile)
    hdulist.info()
    
    print hdulist[1].columns.names
    print hdulist["_TRUTH"].columns.names
    res=hdulist[1].data
    truth=hdulist["_TRUTH"].data
    n=min(res.size,truth.size)
    res=res[:][:n]
    truth=truth[:][:n]
    
    bestz=res["Z"]
    errz=res["ZERR"]
    
    truez=truth["TRUEZ"]
    n=min(bestz.size,truez.size)
    deltaz=bestz-truez
    errz=errz
    truez=truez
    dchi2=res["SECOND_CHI2"]-res["BEST_CHI2"]
    
    ok=np.where(np.abs(deltaz)<0.005)[0]
    print "based on |dz|<0.005 :"
    print "==========================="
    print "efficiency tot = %d/%d = %f"%(ok.size,n,ok.size/float(n))
    print "rms  = %f"%np.std(deltaz[ok])
    print "bias = %f +- %f"%(np.mean(deltaz[ok]),np.std(deltaz[ok])/np.sqrt(ok.size))
    
    dchi2min=40
    ok=np.where(dchi2>dchi2min)[0]
    print "based on dchi2>%d :"%int(dchi2min)
    print "==========================="
    print "efficiency tot = %d/%d = %f"%(ok.size,n,ok.size/float(n))
    print "rms  = %f"%np.std(deltaz[ok])
    print "bias = %f +- %f"%(np.mean(deltaz[ok]),np.std(deltaz[ok])/np.sqrt(ok.size))
    nbad=np.sum(np.abs(deltaz[ok])>0.005)
    print "catastrophic failure rate = %d/%d = %f"%(nbad,ok.size,nbad/float(ok.size))
    print ""
    
    #pylab.figure()
    #pylab.plot(truth["OIIFLUX"],"o")
               
    pylab.figure()
    nx=3
    ny=2
    ai=1
    a=pylab.subplot(ny,nx,ai); ai +=1
    #a.plot(truez,deltaz,"o")
    a.errorbar(truez,deltaz,errz,fmt="o")   
    a.errorbar(truez[ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("Redshift")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(res["SECOND_CHI2"]-res["BEST_CHI2"],deltaz,errz,fmt="o")   
    a.set_xlabel("dchi2")
    a.set_ylabel("Best - True Redshift")

    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(truth["VDISP"],deltaz,errz,fmt="o")   
    a.set_xlabel("vdisp (km/s)")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(res["BEST_SNR"],deltaz,errz,fmt="o")   
    a.set_xlabel("S/N")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(np.log10(truth["OIIFLUX"]),deltaz,errz,fmt="o")   
    a.set_xlabel("log10(OIIflux)")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(res["BEST_AMP_00"],deltaz,errz,fmt="o")   
    #a.plot(deltaz,"o")   
    a.set_xlabel("spec")
    a.set_ylabel("Best - True Redshift")
    
    if False :
        pylab.figure()
        ok=np.where(np.abs(deltaz)<0.005)[0]
        bad=np.where(np.abs(deltaz)>0.005)[0]
        nx=3
        ny=2
        ai=1
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_00"][bad],res["BEST_AMP_01"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_00"][ok],res["BEST_AMP_01"][ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_00"][bad],res["BEST_AMP_02"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_00"][ok],res["BEST_AMP_02"][ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_00"][bad],res["BEST_AMP_03"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_00"][ok],res["BEST_AMP_03"][ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_02"][bad],res["BEST_AMP_03"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_02"][ok],res["BEST_AMP_03"][ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_02"][bad],res["BEST_AMP_04"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_02"][ok],res["BEST_AMP_04"][ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(res["BEST_AMP_02"][bad],res["BEST_AMP_05"][bad],"o",c="b")
        pylab.plot(res["BEST_AMP_02"][ok],res["BEST_AMP_05"][ok],"o",c="r")


    pylab.show()


if __name__ == '__main__':
    main()
    

