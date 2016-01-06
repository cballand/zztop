#!/usr/bin/env python

from astropy.io import fits
import argparse
import pylab
import numpy as np
from math import *

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path to zzbest.fits file')
    args = parser.parse_args()
    
    hdulist=fits.open(args.infile)
    #hdulist.info()
    
    #print hdulist[1].columns.names
    #print hdulist["_TRUTH"].columns.names
    res=hdulist[1].data
    truth=hdulist["_TRUTH"].data
    n=min(res.size,truth.size)
    res=res[:][:n]
    truth=truth[:][:n]
    truez=truth["TRUEZ"]
    bestz=res["Z"]
    errz=res["ZERR"]
    
    if True :
        # try to see if better solutions exist in range of results
        ntrack=8
        allz=np.zeros((ntrack,n))
        allz[0]=res["BEST_Z"]
        allz[1]=res["SECOND_Z"]
        allz[2]=res["THIRD_Z"]
        for i in range(4,ntrack) :
            allz[i]=res["%dTH_Z"%i]
        bbestz=np.zeros((n))
        for i in range(n) :
           bbestz[i]= allz[np.argmin(np.abs(truez[i]-allz[:,i])),i]
        good=np.where(np.abs(bbestz-truez)<0.005)[0]
        print "based on |dz|<0.005 BEST AMONG ALL SAVED SOLUTIONS :"
        print "===================================================="
        print "efficiency tot = %d/%d = %f"%(good.size,n,good.size/float(n))
        print ""
    
    
    n=min(bestz.size,truez.size)
    deltaz=bestz-truez
    errz=errz
    truez=truez
    dchi2=res["SECOND_CHI2"]-res["BEST_CHI2"]
    oIIflux=(res["BEST_FLUX_3727A"]+res["BEST_FLUX_3729A"])

    good=np.where(np.abs(deltaz)<0.005)[0]
    print "based on |dz|<0.005 (after default choice of best solution):"
    print "==========================="
    print "efficiency tot = %d/%d = %f"%(good.size,n,good.size/float(n))
    print "rms  = %f"%np.std(deltaz[good])
    print "bias = %f +- %f"%(np.mean(deltaz[good]),np.std(deltaz[good])/np.sqrt(good.size))
    print ""
    
    dchi2min=0 # 
    snrmin=0 # 
    oIIfluxmin=5.

    ok=np.where((res["BEST_SNR"]>snrmin)&(oIIflux>oIIfluxmin))[0]
    print "based on dchi2>%d and SNR>%f and oIIflux>%f :"%(dchi2min,snrmin,oIIfluxmin)
    print "==========================="
    print "efficiency tot = %d/%d = %f"%(ok.size,n,ok.size/float(n))
    bad=np.where(np.abs(deltaz[ok])>0.005)[0]
    nbad=bad.size
    print "catastrophic failure rate = %d/%d = %f"%(nbad,ok.size,nbad/float(ok.size))
    print "bad fits :"
    for b in bad :
        print "spec=",ok[b],"dz=",deltaz[ok[b]],"dchi2=",dchi2[ok[b]]
    
    
    #pylab.figure()
    #pylab.plot(truth["OIIFLUX"],"o")
    pylab.figure()
    nx=2
    ny=1
    ai=1
    a=pylab.subplot(ny,nx,ai); ai +=1
    a.plot(truth["VDISP"],res["BEST_VDISP"],"o",c="b")
    a.plot(truth["VDISP"][good],res["BEST_VDISP"][good],"o",c="r")
    a.plot(truth["VDISP"],truth["VDISP"],"-",c="gray")
    a.set_xlabel("True vdisp")
    a.set_ylabel("Best vdisp")
    a=pylab.subplot(ny,nx,ai); ai +=1
    oIIflux=(res["BEST_FLUX_3727A"]+res["BEST_FLUX_3729A"])*1e-17
    oIIerr=np.sqrt(res["BEST_FLUX_ERR_3727A"]**2+res["BEST_FLUX_ERR_3729A"]**2)*1e-17
    a.plot(truth["OIIFlux"],oIIflux,"o",c="b")
    #a.plot(truth["OIIFlux"][good],oIIflux[good],"o",c="r")
    a.errorbar(truth["OIIFlux"][good],oIIflux[good],oIIerr[good],fmt="o",color="r")
    print "rms(delta(flux)/err)=",np.std((oIIflux[good]-truth["OIIFlux"][good])/oIIerr[good])
    
    a.plot(truth["OIIFlux"],truth["OIIFlux"],"-",c="gray")
    a.set_xlabel("True OII flux")
    a.set_ylabel("Best OII flux")
    
    
    pylab.figure()
    nx=3
    ny=3
    ai=1
    
    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(truez,deltaz,errz,fmt="o")   
    a.errorbar(truez[ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("True Redshift")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(bestz,deltaz,errz,fmt="o")   
    a.errorbar(bestz[ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("Best Redshift")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(res["SECOND_CHI2"]-res["BEST_CHI2"],deltaz,errz,fmt="o")   
    a.errorbar(res["SECOND_CHI2"][ok]-res["BEST_CHI2"][ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("dchi2")
    a.set_ylabel("Best - True Redshift")

    a=pylab.subplot(ny,nx,ai); ai +=1
    a.errorbar(truth["VDISP"],deltaz,errz,fmt="o")   
    a.errorbar(truth["VDISP"][ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("vdisp (km/s)")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(res["BEST_SNR"],deltaz,errz,fmt="o")   
    a.errorbar(res["BEST_SNR"][ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    a.set_xlabel("S/N")
    a.set_ylabel("Best - True Redshift")
    
    
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(res["BEST_FLUX_3727A"]+res["BEST_FLUX_3729A"],deltaz,errz,fmt="o")   
    a.errorbar(res["BEST_FLUX_3727A"][ok]+res["BEST_FLUX_3729A"][ok],deltaz[ok],errz[ok],fmt="o",c="r")   
    #a.plot(deltaz,"o")   
    a.set_xlabel("meas. OII flux")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(res["BEST_CHI2PDF"],deltaz,errz,fmt="o")   
    a.errorbar(res["BEST_CHI2PDF"][ok],deltaz[ok],errz[ok],fmt="o",c="r")
    a.set_xlabel("best chi2pdf")
    a.set_ylabel("Best - True Redshift")
    
    """
    a=pylab.subplot(ny,nx,ai); ai +=1 
    a.errorbar(np.log10(truth["OIIFLUX"]),deltaz,errz,fmt="o")   
    a.set_xlabel("true log10(OIIflux)")
    a.set_ylabel("Best - True Redshift")
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    ok=np.where((np.abs(deltaz)<0.005)&(res["BEST_FLUX_3727A"]>0))[0]
    ratio=res["BEST_FLUX_3729A"][ok]/res["BEST_FLUX_3727A"][ok]
    a.plot(truth["OIIDOUBLET"][ok],ratio,"o")   
    #a.plot(deltaz,"o")   
    a.set_xlabel("true OII doublet")
    a.set_ylabel("meas. OII doublet")
    

    a=pylab.subplot(ny,nx,ai); ai +=1 
    ok=np.where(np.abs(deltaz)<0.005)[0]
    oIIflux=(res["BEST_FLUX_3727A"]+res["BEST_FLUX_3729A"])[ok]*1e-17
    oIIfluxerr=np.sqrt(res["BEST_FLUX_ERR_3727A"]**2+res["BEST_FLUX_ERR_3729A"]**2)[ok]*1e-17
    a.errorbar(truth["OIIFLUX"][ok],oIIflux,oIIfluxerr,fmt="o")
    """
    
    #a.plot(deltaz,"o")   
    a.set_xlabel("true OII flux")
    a.set_ylabel("meas. OII flux")
    
    
    
    a=pylab.subplot(ny,nx,ai); ai +=1 
    
    nlines_above_nsig=np.zeros((res["BEST_FLUX_3727A"].size))
    snr=np.zeros((res["BEST_FLUX_3727A"].size))
    lines=[3727,3729,4862,4960,5008,6564]
    for line in lines :
        flux=res["BEST_FLUX_%dA"%line]
        err=res["BEST_FLUX_ERR_%dA"%line]
        nlines_above_nsig += ((err>0)*(flux/(err+(err==0)))>2.)
        snr += ((err>0)*(flux/(err+(err==0))))**2
    snr=np.sqrt(snr)
    
    #a.errorbar(nlines_above_nsig,deltaz,errz,fmt="o")
    #a.errorbar(nlines_above_nsig[ok],deltaz[ok],errz[ok],fmt="o",c="r")
    a.plot(bestz,nlines_above_nsig,"o")
    a.plot(bestz[ok],nlines_above_nsig[ok],"o",c="r")
    #a.plot(snr[ok],res["BEST_SNR"][ok],"o",c="r")
    
    #a.plot(deltaz,"o")   
    a.set_ylabel("n lines above 1 sig.")
    a.set_xlabel("Best Redshift")
    
    

    if False :
        pylab.figure()
        rflux=res["BEST_FLUX_3727A"]
        
        nx=3
        ny=2
        ai=1
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(rflux[:],res["BEST_FLUX_3729A"][:]/rflux[:],"o",c="b")
        pylab.plot(rflux[ok],res["BEST_FLUX_3729A"][ok]/rflux[ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(rflux[:],res["BEST_FLUX_4862A"][:]/rflux[:],"o",c="b")
        pylab.plot(rflux[ok],res["BEST_FLUX_4862A"][ok]/rflux[ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(rflux[:],res["BEST_FLUX_4960A"][:]/rflux[:],"o",c="b")
        pylab.plot(rflux[ok],res["BEST_FLUX_4960A"][ok]/rflux[ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(rflux[:],res["BEST_FLUX_5008A"][:]/rflux[:],"o",c="b")
        pylab.plot(rflux[ok],res["BEST_FLUX_5008A"][ok]/rflux[ok],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        pylab.plot(rflux[:],res["BEST_FLUX_6564A"][:]/rflux[:],"o",c="b")
        pylab.plot(rflux[ok],res["BEST_FLUX_6564A"][ok]/rflux[ok],"o",c="r")
        
    if False :
        pylab.figure()
        rflux=res["BEST_FLUX_3729A"]
        nonnull=(rflux>0)
        nx=3
        ny=3
        ai=1
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(rflux[nonnull],res["BEST_FLUX_3727A"][nonnull],"o",c="b")
        a.plot(rflux[good],res["BEST_FLUX_3729A"][good],"o",c="r")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_3727A"][nonnull]/rflux[nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_3727A"][good]/rflux[good],deltaz[good],"o",c="r")
        a.set_xlabel("3727/3729")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_4862A"][nonnull]/rflux[nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_4862A"][good]/rflux[good],deltaz[good],"o",c="r")
        a.set_xlabel("4862/3729")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_4960A"][nonnull]/rflux[nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_4960A"][good]/rflux[good],deltaz[good],"o",c="r")
        a.set_xlabel("4960/3729")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_5008A"][nonnull]/rflux[nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_5008A"][good]/rflux[good],deltaz[good],"o",c="r")
        a.set_xlabel("5008/3729")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_4960A"][nonnull]/res["BEST_FLUX_4862A"][nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_4960A"][good]/res["BEST_FLUX_4862A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("4960/4862")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_5008A"][nonnull]/res["BEST_FLUX_4862A"][nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_5008A"][good]/res["BEST_FLUX_4862A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("5008/4862")        
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_4960A"][nonnull]/res["BEST_FLUX_4862A"][nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_4960A"][good]/res["BEST_FLUX_4862A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("4960/4862")
        
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_6564A"][nonnull]/rflux[nonnull],deltaz[nonnull],"o",c="b")
        a.plot(res["BEST_FLUX_6564A"][good]/rflux[good],deltaz[good],"o",c="r")
        a.set_xlabel("6564/3729")
    if True :
        pylab.figure()
        
        nx=3
        ny=3
        ai=1
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_3729A"][:]/res["BEST_FLUX_4862A"][:],deltaz[:],"o",c="b")
        a.plot(res["BEST_FLUX_3729A"][good]/res["BEST_FLUX_4862A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("3729/4862")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_3729A"][:]/res["BEST_FLUX_4960A"][:],deltaz[:],"o",c="b")
        a.plot(res["BEST_FLUX_3729A"][good]/res["BEST_FLUX_4960A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("3729/4960")
        a=pylab.subplot(ny,nx,ai); ai +=1
        a.plot(res["BEST_FLUX_3729A"][:]/res["BEST_FLUX_5008A"][:],deltaz[:],"o",c="b")
        a.plot(res["BEST_FLUX_3729A"][good]/res["BEST_FLUX_5008A"][good],deltaz[good],"o",c="r")
        a.set_xlabel("3729/5008")
        

    pylab.show()


if __name__ == '__main__':
    main()
    

