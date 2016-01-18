#!/usr/bin/env python

from astropy.io import fits
import argparse
import numpy as np
import re
import string
import sys

from desispec.log import get_logger

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path to zzbest.fits file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path to output json file')
    args = parser.parse_args()
    log  = get_logger()
    
    hdulist=fits.open(args.infile)

    # find list of lines from table keywords
    keys=hdulist[1].columns.names
    table=hdulist[1].data
    
    lines=[]
    for k in keys :
        if k.find("BEST_FLUX_")==0 and  k.find("BEST_FLUX_ERR")<0 :
            numbers=re.findall(r'\d+',k)
            if len(numbers)==1 :
                lines.append(string.atoi(numbers[0]))
    lines=np.unique(np.array(lines))
    log.info("lines in file: %s"%str(lines))

    


    
    oIIline1=3727
    oIIline2=3729
    try :
        oIIflux=table["BEST_FLUX_%dA"%oIIline1]+table["BEST_FLUX_%dA"%oIIline2]
        oIIerr=np.sqrt(table["BEST_FLUX_%dA"%oIIline1]**2+table["BEST_FLUX_%dA"%oIIline2]**2)
    except KeyError :
        log.error("cannot compute oII flux")
        log.error(sys.exc_info())
        sys.exit(12)
    
    # first step : compute an average set of line ratios
    # by scaling fluxes wrt to oII
    # we will then use this average set of line ratios to normalize 
    # all entries and then start the pca
    selection=np.where((oIIflux>0)&(oIIerr>0))[0]
    if selection.size == 0 :
        log.error("no entry with valid oII flux")
        sys.exit(12)
    
     
    flux=np.zeros((selection.size,lines.size))
    ivar=np.zeros((selection.size,lines.size))
    for i in range(lines.size) :
        flux[:,i]=table["BEST_FLUX_%dA"%lines[i]][selection]/oIIflux[selection]
        var=(table["BEST_FLUX_ERR_%dA"%lines[i]][selection]/oIIflux[selection])**2
        # account for error on oIIflux
        var += (flux[:,i]*oIIerr[selection]/oIIflux[selection])**2        
        mask=np.where(var>0)[0]
        ivar[mask,i]=1./var[mask]
    
    # this is the mean line ratios
    sivar=np.sum(ivar,axis=0)
    ok=np.where(sivar>0)[0]
    lines=lines[ok]
    mean_flux_wrt_oII=np.sum(ivar*flux,axis=0)[ok]/sivar[ok]
    err_flux_wrt_oII=1./np.sqrt(sivar[ok])
    
    # refit the amp of each galaxy wrt to mean_flux_wrt_oII
    ngal=table.size
    log.info("number of galaxies = %d"%ngal)
    
    
    # fill array
    flux=np.zeros((ngal,lines.size))
    ivar=np.zeros((ngal,lines.size))
    for i in range(lines.size) :
        flux[:,i]=table["BEST_FLUX_%dA"%lines[i]]
        var=(table["BEST_FLUX_ERR_%dA"%lines[i]])**2
        ok=np.where(var>0)[0]
        ivar[ok,i]=1./var[ok]
    
    # for each gal, fit scale and apply it
    a=np.sum(ivar*mean_flux_wrt_oII**2,axis=1)
    b=np.sum(ivar*mean_flux_wrt_oII*flux,axis=1)
    scale=b/(a+(a==0))
    
    for i in range(ngal) :
        if scale[i] > 0 :
            flux[i] /= scale[i]
            ivar[i] *= scale[i]**2
        else :
            flux[i]=0.
            ivar[i]=0.

    
    
    a    = np.sum(ivar,axis=0)
    mean = np.sum(ivar*flux,axis=0)/a
    
    residuals=flux-mean
    
    # now we can try to do some sort of npca
    eigenvectors=np.zeros((lines.size,lines.size))
    coefs=np.zeros((ngal,lines.size))
    
    bb=np.zeros((lines.size))
    aa=np.zeros((lines.size))
    
    for e in range(lines.size) :
        i=np.argmax(np.std(residuals,axis=0))
        eigenvectors[e]=np.ones(lines.size) # 
        eigenvectors[e]  /= np.sqrt(np.sum(eigenvectors[e]**2))
        
        for loop in range(100) :            
            a=np.sum(ivar*eigenvectors[e]**2,axis=1)
            b=np.sum(ivar*eigenvectors[e]*residuals,axis=1)
            coefs[:,e]=b/(a+(a==0))
            aa *= 0.
            bb *= 0.            
            for i in range(lines.size) : 
                bb[i]=np.sum(ivar[:,i]*coefs[:,e]*residuals[:,i])
                aa[i]=np.sum(ivar[:,i]*coefs[:,e]**2)
            newvect=bb/aa

            # orthogonalize         
            for i in range(e) :
                prod=np.inner(newvect,eigenvectors[i])
                newvect -= eigenvectors[i]
            # normalize
            newvect /= np.sqrt(np.sum(newvect**2))
                        
            dist=np.max(np.abs(newvect-eigenvectors[e]))
            eigenvectors[e]=newvect
            if dist<1e-6 :
                break
        for i in range(lines.size) : 
            residuals[:,i] -= coefs[:,e]*eigenvectors[e,i]
        
    file=open(args.outfile,"w")
    file.write('"pca":{\n')
    file.write('"lines": [')
    for l in lines :
        if l != lines[0] :
            file.write(",")
        file.write("%d"%l)
    file.write('],\n')
    file.write('"mean_flux": [')
    for  e in range(eigenvectors.shape[0]) :
        if e>0 :
            file.write(",")
        file.write("%f"%mean[e])
    file.write('],\n')
    
    file.write('"components": [\n')
    for e in range(eigenvectors.shape[0]) :
        file.write('[')
        for i in range(eigenvectors.shape[1]) :
            if i>0 :
                file.write(",")
            file.write("%f"%eigenvectors[e,i])
        if e<eigenvectors.shape[0]-1 :
            file.write('],\n')
        else :
            file.write(']\n')
    file.write('],\n')
    file.write('"mean_coef": [')
    for e in range(eigenvectors.shape[0]) :
        if e>0 :
            file.write(",")
        file.write("%f"%np.mean(coefs[:,e]))
    file.write('],\n')
    file.write('"rms_coef": [')
    for e in range(eigenvectors.shape[0]) :
        if e>0 :
            file.write(",")
        file.write("%f"%np.std(coefs[:,e]))
    file.write(']\n')
    file.write('}\n')
    
    file.close()
    log.info("wrote %s"%args.outfile)
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
