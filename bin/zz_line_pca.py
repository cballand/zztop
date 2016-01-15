#!/usr/bin/env python

from astropy.io import fits
import argparse
import pylab
import numpy as np
from math import *
import re
import string
import sys

from desispec.log import get_logger

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path to zzbest.fits file')
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

    


    # I require oII flux being measured
    oIIline1=3727
    oIIline2=3729
    """
    i1=np.where(lines==oIIline1)[0]
    i2=np.where(lines==oIIline2)[0]
    if i1.size==0 or i2.size==0 :
       log.error("cannot find oII lines in file")
       sys.exit(12) 
    i1=i1[0]
    i2=i2[0]
    """
    
    
    
    try :
        oIIflux=table["BEST_FLUX_%dA"%oIIline1]+table["BEST_FLUX_%dA"%oIIline2]
        oIIerr=np.sqrt(table["BEST_FLUX_%dA"%oIIline1]**2+table["BEST_FLUX_%dA"%oIIline2]**2)
    except KeyError :
        log.error("cannot compute oII flux")
        log.error(sys.exc_info())
        sys.exit(12)
    
     
    selection=np.where((oIIflux>0)&(oIIerr>0))[0]
    if selection.size == 0 :
        log.error("no entry with valid oII flux")
        sys.exit(12)
    
    # remove oII doublet from list of lines for pca
    # and check we have data
    lines_for_pca=[]
    
    for line in lines :
        if line==oIIline1 or line==oIIline2 :
            continue
        nmeas=np.sum(table["BEST_FLUX_ERR_%dA"%line][selection]>0)
        if nmeas==0 :
            log.warning("no valid measurement for line %dA"%line)
            continue
        lines_for_pca.append(line)
    lines_for_pca=np.array(lines_for_pca)
    
    flux=np.zeros((selection.size,lines_for_pca.size))
    ivar=np.zeros((selection.size,lines_for_pca.size))
    for i in range(lines_for_pca.size) :
        flux[:,i]=table["BEST_FLUX_%dA"%lines_for_pca[i]][selection]/oIIflux[selection]
        var=(table["BEST_FLUX_ERR_%dA"%lines_for_pca[i]][selection]/oIIflux[selection])**2
        # account for error on oIIflux
        var += (flux[:,i]*oIIerr[selection]/oIIflux[selection])**2        
        mask=np.where(var>0)[0]
        ivar[mask,i]=1./var[mask]

    mean=np.sum(ivar*flux,axis=0)/np.sum(ivar,axis=0)
    
    
    residuals=flux-mean
    
    # now we can try to do some sort of npca
    eigenvectors=np.zeros((lines_for_pca.size,lines_for_pca.size))
    coefs=np.zeros((selection.size,lines_for_pca.size))
    
    bb=np.zeros((lines_for_pca.size))
    aa=np.zeros((lines_for_pca.size))
    
    for e in range(lines_for_pca.size) :
        i=np.argmax(np.std(residuals,axis=0))
        eigenvectors[e]=np.ones(lines_for_pca.size) # 
        eigenvectors[e]  /= np.sqrt(np.sum(eigenvectors[e]**2))
        
        for loop in range(100) :            
            a=np.sum(ivar*eigenvectors[e]**2,axis=1)
            b=np.sum(ivar*eigenvectors[e]*residuals,axis=1)
            coefs[:,e]=b/(a+(a==0))
            aa *= 0.
            bb *= 0.            
            for i in range(lines_for_pca.size) : 
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
            #print loop,":",eigenvectors[e]," dist=",dist
            if dist<1e-6 :
                break
        # remove this component
        for i in range(lines_for_pca.size) : 
            residuals[:,i] -= coefs[:,e]*eigenvectors[e,i]
        

    for e in range(lines_for_pca.size) : 
        log.info("coef #%d mean= %f rms= %f"%(e,np.mean(coefs[:,e]),np.std(coefs[:,e])))
        h,b=np.histogram(coefs[:,e],bins=50)
        x=b[:-1]+(b[1]-b[0])/2.
        pylab.plot(x,h,label="comp #%d"%e)
    log.info("mean ratios = %s"%str(mean))
    for e in range(lines_for_pca.size) :
        log.info("comp #%d = %s norme=%f"%(e,str(eigenvectors[e]),np.sum(eigenvectors[e]**2)))
    
    pylab.legend(loc="upper right")












    pylab.show()
    sys.exit(12)
        
        
        

    pylab.plot(lines_for_pca,mean,"o")
    pylab.show()
    
    log.info("lines for pca: %s"%str(lines_for_pca))
    
    
if __name__ == '__main__':
    main()
