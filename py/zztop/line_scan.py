from desispec.log import get_logger
from desispec.linalg import cholesky_solve

import numpy as np
import scipy.sparse
import math
import sys
import time

def find_groups(lines,restframe_sigmas,resolution_sigma=1.,nsig=5.) :
    """
    internal routine : find group of lines that we have to fit together because they overlap
    """
    line_group=-1*np.ones((lines.size)).astype(int)
    line_group[0]=0 # add first one
    
    for line_index in range(lines.size) :
        line=lines[line_index]
        for group_index in np.where(line_group>=0)[0] :
            for other_line in lines[np.where(line_group==group_index)[0]] :
                if (line-other_line)**2 < nsig**2*(restframe_sigmas[line_index]**2+resolution_sigma**2) :
                    line_group[line_index]=group_index
                    break
            if line_group[line_index]>=0 :
                break
        if line_group[line_index]==-1 :
            line_group[line_index]=np.max(line_group)+1
        
    
    groups={}
    for g in np.unique(line_group) :
        groups[g]=np.where(line_group==g)[0]
    
    
    return groups

def zz_line_fit(wave,flux,ivar,resolution,lines,vdisp,line_ratio_priors,z,wave_nsig,x,gx,groups,fixed_line_ratio=None) :
    """
    internal routine : fit line amplitudes and return delta chi2 with respect to zero amplitude
    """
    log=get_logger()

    #z=1.334595
    #z=0.79
    show=False
    
    
    redshifted_lines=lines*(1+z)
    redshifted_sigmas=lines*(vdisp/2.9970e5)*(1+z) # this is the sigmas of all lines
    
    delta_chi2_coeff_per_group={} # either a scalar or a vector per group (it's getting complicated)

    nframes=len(flux)
    
    # compute profiles, and fill A and B matrices
    # A and B are matrix and vector so that line amplitudes are solution of A*amp = B
    
    A=np.zeros((lines.size,lines.size))
    B=np.zeros((lines.size))
    
    # do it per group to account for overlapping lines
    for group_index in groups :
        lines_in_group=groups[group_index]
        l1=np.min(redshifted_lines[lines_in_group]-wave_nsig*redshifted_sigmas[lines_in_group])
        l2=np.max(redshifted_lines[lines_in_group]+wave_nsig*redshifted_sigmas[lines_in_group])            
        nlines=lines_in_group.size
                    
        for frame_index in range(nframes) :
            frame_wave = wave[frame_index]
            frame_ivar = ivar[frame_index]
            nw=frame_wave.size
            # wavelength that matter :
            wave_index=np.where((frame_wave>=l1)&(frame_wave<=l2)&(frame_ivar>0))[0]
            if wave_index.size == 0 :
                continue
            frame_wave=frame_wave[wave_index]
            frame_ivar=frame_ivar[wave_index]
            frame_flux=flux[frame_index][wave_index]
                
            # this is the block of the diagonal sparse matrix corresponding to the wavelength of interest :
            frame_res_for_group=scipy.sparse.dia_matrix((resolution[frame_index].data[:,wave_index], resolution[frame_index].offsets), shape=(wave_index.size, wave_index.size))
            
            
            
                
            # compute profiles    
            profile_of_lines=np.zeros((lines_in_group.size,frame_wave.size))
            for i,line_index,line,sig in zip(range(lines_in_group.size),lines_in_group,redshifted_lines[lines_in_group],redshifted_sigmas[lines_in_group]) :

                # simple Gaussian given velocity dispersion, correcty normalized to give integrated flux
                prof=np.interp((frame_wave-line)/sig,x,gx)/sig
                
                # convolve here with the spectrograph resolution
                profile_of_lines[i]=frame_res_for_group.dot(prof)
            
            if show and group_index==0 :
                print "DEBUGGING !!!"
                w=frame_wave
                f=frame_flux
                p0=profile_of_lines[0]
                p1=profile_of_lines[1]
            
            

            # fill amplitude system (A and B) :
            for i in range(nlines) :
                B[lines_in_group[i]]   += np.sum(frame_ivar*profile_of_lines[i]*frame_flux)
                for j in range(nlines) :
                    A[lines_in_group[i],lines_in_group[j]] += np.sum(frame_ivar*profile_of_lines[i]*profile_of_lines[j])
                    
            
            
    
    
    
    # solving outside of group to simplify the code even if it is supposedly slower (the matrix is nearly diagonal)
    line_amplitudes_ivar = np.diag(A)
    
    
    if fixed_line_ratio is not None :
        
        for i in fixed_line_ratio :
            j=fixed_line_ratio[i][0]
            ratio=fixed_line_ratio[i][1]
            #log.debug("fixed ratio %d/%d = %f"%(lines[i],lines[j],ratio))
            # f0=ratio*f1 (f0<f1)
            # dmd0' = dmd0*(d0/d0') + dmd1*(d1/d0) =  dmd0 + dmd1*(1/ratio)
            # aij = sum w(dmdi)(dmdj)
            # a00' propto (dmd0')**2 = (dmd0 + dmd1/ratio)**2 propto a00+a11/ratio**2+2*a01/ratio
            # b0'  propto  dmd0' = b0 + b1/ratio
            # variable 1 disappears
            
            B[i]   += B[j]/ratio
            A[i,i] += 1./ratio**2*A[j,j] + 2./ratio*A[i,j]
            A[i,j] = 0
            A[j,i] = 0
            B[j]   = 0
        # leave A[j,j] that wont affect amp0
    
    # give value to undefined lines (it's not a problem)
    for i in range(lines.size) :
        if A[i,i]==0 :
            A[i,i]=1

    # solve the system
    try :
        line_amplitudes = cholesky_solve(A,B)
    except  :
        log.warning("cholesky_solve failed")
        return 1e5,np.zeros((lines.size)),np.zeros((lines.size))
    
    if fixed_line_ratio is not None :
        
        for i in fixed_line_ratio :
            j=fixed_line_ratio[i][0]
            ratio=fixed_line_ratio[i][1]
            line_amplitudes[j]=line_amplitudes[i]/ratio
        
        
    if show :
        import pylab
        pylab.plot(w,f)
        pylab.plot(w,line_amplitudes[0]*p0)
        pylab.plot(w,line_amplitudes[1]*p1)
        pylab.plot(w,line_amplitudes[0]*p0+line_amplitudes[1]*p1)
        print "DEBUGGING!!"
        pylab.show()
    
    # apply priors (outside of loop on groups)
    if line_ratio_priors is not None :
        for line_index in line_ratio_priors :
            
            other_line_index = int(line_ratio_priors[line_index][0])
            min_ratio        = float(line_ratio_priors[line_index][1])
            max_ratio        = float(line_ratio_priors[line_index][2])
            conserve_flux    = line_ratio_priors[line_index][3]
            
            # first ignore if one of the lines is not measured
            if line_amplitudes_ivar[line_index]==0 or line_amplitudes_ivar[other_line_index]==0 :
                continue

            # if two lines are negatives ignore this
            if line_amplitudes[line_index]<=0 and line_amplitudes[other_line_index]<=0 :
                continue
            
            # the ratio prior is on flux(this_line)/flux(other_line)
            if conserve_flux :
                
                total_flux = line_amplitudes[line_index]+line_amplitudes[other_line_index]
                
                if line_amplitudes[other_line_index]<=0 :
                    ratio = 10000.
                else :
                    ratio = line_amplitudes[line_index]/line_amplitudes[other_line_index]
                
                if ratio > max_ratio :
                    line_amplitudes[line_index] = max_ratio/(1.+max_ratio)*total_flux
                    line_amplitudes[other_line_index] = 1./(1.+max_ratio)*total_flux
                elif ratio < min_ratio :
                    line_amplitudes[line_index] = min_ratio/(1.+min_ratio)*total_flux
                    line_amplitudes[other_line_index] = 1./(1.+min_ratio)*total_flux
            else :
                # apply ratio to line with lowest snr :
                this_snr=line_amplitudes[line_index]*math.sqrt(line_amplitudes_ivar[line_index])
                other_snr=line_amplitudes[other_line_index]*math.sqrt(line_amplitudes_ivar[other_line_index])
            
                if this_snr < other_snr :
                    apply_to_index = line_index
                    ratio = line_amplitudes[line_index]/line_amplitudes[other_line_index]
                    if ratio<min_ratio :                    
                        line_amplitudes[line_index]  = line_amplitudes[other_line_index]*min_ratio
                    elif ratio>max_ratio :
                        line_amplitudes[line_index]  = line_amplitudes[other_line_index]*max_ratio
                else :
                    apply_to_index = other_line_index
                    # and need to invert ratio
                    min_ratio_tmp = 1./(max_ratio)
                    max_ratio     = 1./(min_ratio+0.001*(min_ratio==0))
                    min_ratio     = min_ratio_tmp
                    ratio = line_amplitudes[other_line_index]/line_amplitudes[line_index]
                    if ratio<min_ratio :                    
                        line_amplitudes[other_line_index]  = line_amplitudes[line_index]*min_ratio
                    elif ratio>max_ratio :
                        line_amplitudes[other_line_index]  = line_amplitudes[line_index]*max_ratio
    
    # force non-negative here ?
    line_amplitudes[line_amplitudes<0]=0.
    
    # add chi2 for this group
    """
    chi2 = sum w*(data - amp*prof)**2
    = sum w*data**2 + amp**2 sum w prof**2 - 2 * amp * sum w*data*prof
    = chi20 + amp^T A amp - 2 B^T amp
    
    min chi2 for amp = A^-1 B
    min chi2 = chi20 - B^T A^-1 B
    
    BUT, we may have changed the amplitudes with the prior !
    """
    
    # apply delta chi2 per group
    dchi2 = A.dot(line_amplitudes).T.dot(line_amplitudes) - 2*np.inner(B,line_amplitudes)
    
    # return
    return dchi2,line_amplitudes,line_amplitudes_ivar

class SolutionTracker :
    
    def __init__(self) :
        self.zscan = []
        self.chi2scan = []
        
    def find_best(self,ntrack=3,min_delta_z=0.002) :
        # first convert into np arrays
        zscan = np.array(self.zscan)
        chi2scan = np.array(self.chi2scan)
        nfound=0        
        zz=[]
        ze=[]
        cc=[]
        # loop on minima
        while(nfound<ntrack) :
            if chi2scan.size == 0 :
                break
            ibest=np.argmin(chi2scan)
            zbest=zscan[ibest]
            chi2best=chi2scan[ibest]
            zz.append(zbest)
            cc.append(chi2best)
            
            nfound += 1
            
            # estimate error
            dz=0.
            # >z side            
            for i in range(ibest+1,zscan.size) :
                if chi2scan[i]>chi2best+1 :
                    break                 
                dz=abs(zscan[i]-zbest)
            # <z side            
            for i in range(ibest-1,-1,-1) :
                if chi2scan[i]>chi2best+1 :
                    break                 
                dz=max(abs(zbest-zscan[i]),dz)
            ze.append(dz) # can be null
            

            # remove region around local minimum
            mask=np.zeros((zscan.size))
            mask[ibest]=1
            # remove whole local minimum
            # >z side
            previous_chi2=chi2best
            for i in range(ibest+1,zscan.size) :
                if chi2scan[i]>previous_chi2 :
                    mask[i]=1
                    previous_chi2=chi2scan[i]                    
                else :
                    break
            # <z side
            previous_chi2=chi2best
            for i in range(ibest-1,-1,-1) :
                if chi2scan[i]>previous_chi2 :
                    mask[i]=1
                    previous_chi2=chi2scan[i]
                else :
                    break
            # add extra regions
            mask[np.abs(zscan-zbest)<min_delta_z]=1
            # apply mask
            zscan=zscan[mask==0]
            chi2scan=chi2scan[mask==0]
        
        return np.array(zz),np.array(ze),np.array(cc)

    def add(self,z,chi2) :      
        self.zscan.append(z)
        self.chi2scan.append(chi2)

def chi2_of_line_ratio(list_of_results,lines,line_ratio_constraints) :
        
    nres=len(list_of_results)
    chi2=np.zeros((nres))
    log=get_logger()
    for i,result in zip(range(nres),list_of_results) :
        for line_index in line_ratio_constraints :
            line2=int(lines[line_index])
            line1=int(lines[line_ratio_constraints[line_index][0]])
            min_ratio        = float(line_ratio_constraints[line_index][1])
            max_ratio        = float(line_ratio_constraints[line_index][2])
            flux2=result["FLUX_%dA"%line2]
            err2=result["FLUX_ERR_%dA"%line2]
            flux1=result["FLUX_%dA"%line1]
            err1=result["FLUX_ERR_%dA"%line1]
            if err1==0 or err2==0 : # can't tell
                continue
            if flux1<=0 : # tmp
                flux1=err1
            ratio=flux2/flux1
            rerr=math.sqrt((err2/flux1)**2+(flux2*err1/flux1**2)**2)
            log.debug("checking line ratio %dA/%dA  %f<? %f+-%f <? %f"%(line2,line1,min_ratio,ratio,rerr,max_ratio))
            if ratio<min_ratio :
                chi2[i] += (ratio-min_ratio)**2/rerr**2
            elif ratio>max_ratio :
                chi2[i] += (ratio-max_ratio)**2/rerr**2
        log.debug("%d z=%f ratio chi2=%f spec chi2=%f"%(i,result["Z"],chi2[i],result["CHI2"]))
    return chi2
            
            
def zz_subtract_continuum(wave,flux,ivar,wave_step=150.) :
    
    
    # merge all data
    allwave=wave[0]
    allflux=flux[0]
    allivar=ivar[0]
    for index in range(1,len(wave)) :
        allwave=np.append(allwave,wave[index])
        allflux=np.append(allflux,flux[index])
        allivar=np.append(allivar,ivar[index])
    ii=np.argsort(allwave)
    allwave=allwave[ii]
    allflux=allflux[ii]
    allivar=allivar[ii]
    
    
    wmin=np.min(allwave[allivar>0])
    wmax=np.max(allwave[allivar>0])
    n=int((wmax-wmin)/wave_step)    
    knots=wmin+(wmax-wmin)/(n+1)*(0.5+np.arange(n))
    # iterative fitting and clipping (to remove dependence on emission lines!)
    for loop in range(10) : 
        toto=scipy.interpolate.splrep(allwave,allflux,k=3,task=-1,t=knots,w=allivar)
        continuum=scipy.interpolate.splev(allwave,toto)
        out=np.where(allivar*(allflux-continuum)**2>4.)[0]
        if out.size==0 :
            break
        
        allivar[out]=0.
    
    #import pylab
    #pylab.plot(allwave,allflux,"-",c="b")
    #pylab.plot(allwave,continuum,"-",c="r")
    #pylab.plot(allwave,allivar*(allflux-continuum)**2,"-",c="k")
    #pylab.show()
    
    subtracted_flux=[]
    for index in range(len(wave)) :
        subtracted_flux.append(flux[index]-np.interp(wave[index],allwave,continuum))
    return subtracted_flux
            
def zz_line_scan(wave,flux,ivar,resolution,lines,vdisps_fast,vdisps_improved,line_ratio_priors=None,line_ratio_constraints=None,fixed_line_ratio=None,zstep=0.001,zmin=0.,zmax=100.,wave_nsig=3.,ntrack=3,remove_continuum=True,recursive=True) :

    """
    args :
      wave : list of 1D array of wavelength in A
      flux : list of 1D array of flux in ergs/s/cm2/A
      ivar : list of 1D array of flux inverse variance
      resolution : list of resolution matrices
             wave,flux,ivar,resolution lists must have same length
             one must have wave[i].size=flux[i].size ...
      
      lines : 1D array of emission line wavelengths in A
      vdisps : fixed assumed velocity dispersions in km/s for the lines 
             one must have vdisps.size=lines.size
    options :
      line_ratio_priors : dictionnary of priors on line ratio USED IN CHI2
             the key is the index in lines of this_line
             the value is 1D array of size 4 : 
                 array[0] = index of other_line
                 array[1] = min allowed value of ratio flux[this_line]/flux[other_line]
                 array[2] = max allowed value of ratio flux[this_line]/flux[other_line]
                 array[3] = impose total flux conservation (boolean) 
             line_ratio_priors can be set for a limited number of lines
      line_ratio_constraints : dictionnary of priors on line ratio USED FOR RANK BEST SOLUTIONS
             the key is the index in lines of this_line
             the value is 1D array of size 3 : 
                 array[0] = index of other_line
                 array[1] = min allowed value of ratio flux[this_line]/flux[other_line]
                 array[2] = max allowed value of ratio flux[this_line]/flux[other_line]
             line_ratio_constraints can be set for a limited number of lines
      zstep  : step of redshift scan (linear)
      zmin   : force min value of redshift scan (default is automatic)
      zmax   : force max value of redshift scan (default is automatic)
      wave_nsig : defines the range of wavelength used to fit this line (modified in refinement)
      ntrack : number of local minima to track and record (default is 3)
      recursive : internal parameter, one must leave it to True
    
    returns :
       a dictionnary with results of the fit including redshift, uncertainty, chi2, line amplitudes,
       for ntrack best fit local minima. 
    
    
    Some details.
    
    We use the following development to go faster :

    chi2 = sum_frames_f  sum_wave_w  ivar(fw) ( flux(fw) - sum_lines_l a_l prof(lfw) )**2
    
    We group lines that do not overlap (accounting for velocity disp and resolution at 5 sigma) and
    use the following development to go faster :

     chi2 = sum_fw  ivar(fw) ( flux(fw) - sum_groups_g  sum_lines_l a_l prof(lfw) )**2
          = [sum_fw  ivar(fw) flux(fw)**2] - 2 sum_g [ sum_fw  ivar(fw) flux(fw) (sum_l a_l prof(lfw)) ] + sum_g [ sum_fw  ivar(fw) (sum_l a_l prof(lfw))**2 ]
          = chi2_0 + sum_g [ -2 sum_fw  ivar(fw) flux(fw) (sum_l a_l prof(lfw))  sum_fw  ivar(fw) (sum_l a_l prof(lfw))**2 ]
    
    We solve independently for each group of lines, it is fast because each line or group of lines addresses few wavelength
    
    If only one line in the group :
    calling 
    a = sum_fw  ivar(fw) prof(lfw)**2
    b = sum_fw  ivar(fw) flux(fw) prof(lfw)
    the amplitude of the line is x=b/a
    and dchi2 = chi2(amplitude !=0) - chi2(amplitude =0) = -2*b*x + a*x**2 = -b**2/a
    
    If several lines in the group
    A_ij = sum_fw  ivar(fw) prof(ifw) prof(jfw)
    B_i  = sum_fw  ivar(fw) flux(fw) prof(ifw)
    the amplitudes X are given by the solution of A.X = B
    and dchi2 = chi2(X !=0) - chi2(X =0) = -2*B^T X + X^T A X = - B^T X
    
    So, for each redshift in the scan,
    for each group we determine a,b or A,B,X,
    and compute the total chi2 :
    chi2(z) = chi2_0 + sum_g dchi2(g,z)
    
    """
    
    start_time = time.clock( )
    log=get_logger()
    
    nframes=len(wave)
    

    ndata=0
    for index in range(nframes) :
        ndata += np.sum(ivar[index]>0)
    
    if ndata<2 :
        log.warning("ndata=%d skip this spectrum")
        return None,None

    log.debug("lines=%s"%str(lines))
    if line_ratio_priors is not None :
        log.debug("line_ratio_priors=%s"%str(line_ratio_priors))
    log.debug("vdisps_fast=%s km/s"%str(vdisps_fast))
    log.debug("vdisps_improved=%s km/s"%str(vdisps_improved))
    log.debug("zstep=%f"%zstep)
    log.debug("nframes=%d"%nframes)
    log.debug("nlines=%d"%(lines.size))
    
    # find group of lines that we have to fit together because they overlap
    groups = find_groups(lines,lines*np.max(vdisps_fast)/2.9970e5) # consider the largest velocity dispersion here
    
    for g in groups :
        log.debug("group=%d lines=%s"%(g,lines[groups[g]]))

    nframes=len(wave)
    zrange=np.zeros((nframes,2))
    
        
    lmin=np.min(lines)
    lmax=np.max(lines)

    for index in range(nframes) :
        frame_wave=wave[index]
        frame_ivar=ivar[index]
         # define z range
        wmin=np.min(frame_wave[frame_ivar>0])
        wmax=np.max(frame_wave[frame_ivar>0])
        #log.debug("frame #%d wave range =%f %f"%(index,wmin,wmax))
        zrange[index,0]=wmin/lmax-1
        zrange[index,1]=wmax/lmin-1
        log.debug("frame #%d z range to scan=%f %f"%(index,zrange[index,0],zrange[index,1]))
        
    
    zmin=max(zmin,np.min(zrange[:,0]))
    zmax=min(zmax,np.max(zrange[:,1]))
    
    log.debug("zmin=%f zmax=%f zstep=%f"%(zmin,zmax,zstep))
    
    
    # compute one highres gaussian to go faster and not call exp() after
    x=np.linspace(-5.,5.,100)
    gx=1./math.sqrt(2*math.pi)*np.exp(-0.5*x**2)
        
    # create tracker
    tracker = SolutionTracker()

    # fit a continuum and remove it
    if remove_continuum :
        # use spline
        flux_to_fit = zz_subtract_continuum(wave,flux,ivar)
    else :
        flux_to_fit = flux

    # compute chi2 for zero lines
    chi2_0 = 0.
    for frame_index in range(nframes) :
        chi2_0 += np.sum(ivar[frame_index]*flux[frame_index]**2)
        
    line_amplitudes=np.zeros((lines.size))
    line_amplitudes_ivar=np.zeros((lines.size))

    # redshift scan
    best_chi2 = 0
    for z in np.linspace(zmin,zmax,num=int((zmax-zmin)/zstep+1)) :

        # the whole fit happens here
        # do a loop on range of velocity dispersion
        dchi2=1e12
        best_vdisp_for_z=0
        for vdisp in vdisps_fast :
            v_dchi2,v_line_amplitudes,v_line_amplitudes_ivar = zz_line_fit(wave,flux_to_fit,ivar,resolution,lines,vdisp,line_ratio_priors,z,wave_nsig,x,gx,groups,fixed_line_ratio=fixed_line_ratio)
            if v_dchi2 < dchi2 :
                dchi2=v_dchi2
                line_amplitudes = v_line_amplitudes
                line_amplitudes_ivar = v_line_amplitudes_ivar
                best_vdisp_for_z = vdisp
                #log.debug("vdisp=%f dchi2=%f amp=%s"%(best_vdisp_for_z,dchi2,line_amplitudes))
        chi2 = chi2_0 + dchi2
        
        # keep best
        if chi2<best_chi2 or best_chi2==0 :
            best_z_line_amplitudes=line_amplitudes.copy()
            best_z_line_amplitudes_ivar=line_amplitudes_ivar.copy()
            best_vdisp = best_vdisp_for_z
        # now we have to keep track of several solutions
        tracker.add(z=z,chi2=chi2)
    
    log.debug("find_best in range %f %f (nz=%d)"%(tracker.zscan[0],tracker.zscan[-1],len(tracker.zscan)))
    best_zs,best_z_errors,best_chi2s=tracker.find_best(ntrack=ntrack,min_delta_z=0.002)
    best_z_errors[best_z_errors<zstep]=zstep
        
    if recursive :

        log.debug("first pass best z =%f chi2/ndata=%f, second z=%f dchi2=%f, third z=%f dchi2=%f"%(best_zs[0],best_chi2s[0]/ndata,best_zs[1],best_chi2s[1]-best_chi2s[0],best_zs[2],best_chi2s[2]-best_chi2s[0]))

        
        #you want to see the redshift scan for this one ?
        #import pylab
        #pylab.plot(tracker.zscan,tracker.chi2scan)
        #pylab.show()
        #sys.exit(12)
        

        # if recursive we refit here all of the best chi2s
        best_results=[]

        full_zscan = np.array(tracker.zscan)
        full_chi2scan = np.array(tracker.chi2scan)
        

        # define rank label for results
        rank_labels = np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(rank_labels.size,ntrack) :
            rank_labels=np.append(rank_labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
        
        # here we loop on all the tracked solutions
        current_zstep = zstep
        for rank in range(best_zs.size) :
            # second loop about minimum
            # where we save things to compute errors
            dz=current_zstep
            zmin      = best_zs[rank]-dz
            zmax      = best_zs[rank]+dz  
            zstep=(zmax-zmin)/100. # new refined zstep
            log.debug("for rank = %d zmin = %f zmax = %f zstep = %f"%(rank,zmin,zmax,zstep))
            res = zz_line_scan(wave,flux_to_fit,ivar,resolution,lines,vdisps_fast=vdisps_improved,vdisps_improved=None,
                               line_ratio_priors=line_ratio_priors,fixed_line_ratio=fixed_line_ratio,
                               zstep=zstep,zmin=zmin,zmax=zmax,wave_nsig=5.,recursive=False,ntrack=1,remove_continuum=False)
            best_results.append(res)
            
        # now that we have finished improving the fits,
        # order results if it turns out that the ranking has been modified by the improved fit
        chi2=np.zeros((ntrack))
        for i in range(ntrack) :
            chi2[i]=best_results[i]["CHI2PDF"]
        indices=np.argsort(chi2)
        if np.sum(np.abs(indices-range(ntrack)))>0 : # need swap
            swapped_best_results=[]
            for i in range(ntrack) :
                swapped_best_results.append(best_results[indices[i]])
            best_results=swapped_best_results
        
        # if we have line_ratio_constraints, move not satifying solutions to the end
        if line_ratio_constraints is not None :
            chi2_of_ratios=chi2_of_line_ratio(best_results,lines,line_ratio_constraints)
            # use this info ?
            if chi2_of_ratios[0]>0 : # we are outside of the line ratio bounds
                indices=np.argsort(chi2_of_ratios)
                if np.sum(np.abs(indices-range(ntrack)))>0 : # need swap
                    log.warning("SWAPPING results based on chi2 of line ratios : best z %f -> %f"%(best_results[0]["Z"],best_results[indices[0]]["Z"]))
                    swapped_best_results=[]
                    for i in range(ntrack) :
                        swapped_best_results.append(best_results[indices[i]])
                    best_results=swapped_best_results

        labels=np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(labels.size,ntrack) :
            labels=np.append(labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
        final_results={}
        for i in range(ntrack) :
            for k in best_results[i].keys() :
                final_results["%s_%s"%(labels[i],k)]=best_results[i][k]
                
        #log.info("DEBUG DEBUG %f %f %f"%(final_results["BEST_CHI2PDF"],final_results["SECOND_CHI2PDF"],final_results["THIRD_CHI2PDF"]))
           


        end_time = time.clock( )
        log.info("best z=%f+-%f chi2/ndf=%3.2f dchi2=%3.1f time=%f sec"%(final_results["BEST_Z"],final_results["BEST_Z_ERR"],final_results["BEST_CHI2PDF"],final_results["SECOND_CHI2"]-final_results["BEST_CHI2"],end_time-start_time))
        return final_results
    
    # here we are outside of the recursive loop
            
    
    ndf=0
    for index in range(nframes) :
        ndf += np.sum(ivar[index]>0)
    ndf-=(np.sum(best_z_line_amplitudes_ivar>0)+1)
    
    snr=math.sqrt(np.sum(best_z_line_amplitudes**2*best_z_line_amplitudes_ivar))
    

    
    
    res={}
    res["Z"]=best_zs[0]
    res["Z_ERR"]=best_z_errors[0]
    res["CHI2"]=best_chi2s[0]
    res["CHI2PDF"]=best_chi2s[0]/ndf
    res["SNR"]=snr
    res["VDISP"]=best_vdisp
    
    for line_index in range(lines.size) :
        # need to normalize flux by the sigma of the gaussians used
        res["FLUX_%dA"%lines[line_index]]=best_z_line_amplitudes[line_index]
        livar=best_z_line_amplitudes_ivar[line_index]
        res["FLUX_ERR_%dA"%lines[line_index]]=(livar>0)/math.sqrt(livar+(livar==0))
    
    
    
    return res
