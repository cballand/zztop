from desispec.log import get_logger
from desispec.linalg import cholesky_solve

import numpy as np
import scipy.sparse
import math
import sys
import pylab

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

def zz_line_fit(wave,flux,ivar,resolution,lines,restframe_sigmas,line_ratio_priors,z,wave_nsig,x,gx,groups) :
    """
    internal routine : fit line amplitudes and return delta chi2 with respect to zero amplitude
    """
    redshifted_lines=lines*(1+z)
    redshifted_sigmas=restframe_sigmas*(1+z) # this is the sigmas of all lines
    
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

                # simple Gaussian given velocity dispersion
                prof=np.interp((frame_wave-line)/sig,x,gx)
                
                # convolve here with the spectrograph resolution
                profile_of_lines[i]=frame_res_for_group.dot(prof)
            
            # fill amplitude system (A and B) :
            for i in range(nlines) :
                B[lines_in_group[i]]   += np.sum(frame_ivar*profile_of_lines[i]*frame_flux)
                for j in range(nlines) :
                    A[lines_in_group[i],lines_in_group[j]] += np.sum(frame_ivar*profile_of_lines[i]*profile_of_lines[j])
        
    
    # solving outside of group to simplify the code even if it is supposedly slower (the matrix is nearly diagonal)
    line_amplitudes_ivar = np.diag(A)
    
    
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
    
    # apply priors (outside of loop on groups)
    if line_ratio_priors is not None :
        for line_index in line_ratio_priors :
            other_line_index = int(line_ratio_priors[line_index][0])
            min_ratio        = float(line_ratio_priors[line_index][1])
            max_ratio        = float(line_ratio_priors[line_index][2])
            # ratio = this_line/other_line
            norme = np.sum(line_amplitudes_ivar[[line_index,other_line_index]])
            if norme==0 :
                continue   
            total_amplitude   = np.sum((line_amplitudes_ivar*line_amplitudes)[[line_index,other_line_index]])/norme

            if total_amplitude==0 :
                continue
            if line_amplitudes[other_line_index]==0 :
                # use max allowed value
                line_amplitudes[other_line_index] = total_amplitude/(1.+max_ratio)
                line_amplitudes[line_index]       = total_amplitude*max_ratio/(1.+max_ratio)
            else :
                ratio = line_amplitudes[line_index]/line_amplitudes[other_line_index]
                if ratio<min_ratio :
                    line_amplitudes[other_line_index] = total_amplitude/(1.+min_ratio)
                    line_amplitudes[line_index]       = total_amplitude*min_ratio/(1.+min_ratio)
                elif ratio>max_ratio :
                    line_amplitudes[other_line_index] = total_amplitude/(1.+max_ratio)
                    line_amplitudes[line_index]       = total_amplitude*max_ratio/(1.+max_ratio)

    # force non-negative here?
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

        """
        self.ntrack=ntrack
        self.big_number=1e12
        self.best_chi2s     = self.big_number*np.ones((self.ntrack))
        self.best_zs        = np.zeros((self.ntrack))
        self.chi2s_at_best_z_minus_zstep = self.big_number*np.ones((self.ntrack))
        self.chi2s_at_best_z_plus_zstep  = self.big_number*np.ones((self.ntrack))
        self.previous_is_at_rank=-1
        self.previous_chi2  = self.big_number
        self.previous_z     = 0.
        self.chi2_has_increased = True
        self.log=get_logger()
        """
        
    
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

"""
    def estimate_redshift_errors_and_interpolate_best_solutions(self,zstep) :
        # using second derivative of chi2
        best_z_errors = np.zeros((self.ntrack))
        for rank in range(self.best_zs.size) :
            coeffs=np.polyfit([self.best_zs[rank]-zstep,self.best_zs[rank],self.best_zs[rank]+zstep],[self.chi2s_at_best_z_minus_zstep[rank],self.best_chi2s[rank],self.chi2s_at_best_z_plus_zstep[rank]],2)
            a=coeffs[0]
            b=coeffs[1]
            c=coeffs[2]
    
            best_z_errors[rank] = zstep
            if a>0 :
                self.best_zs[rank]       = -b/(2*a)
                self.best_chi2s[rank]    = c-b**2/(4*a)
                best_z_errors[rank] = 1./math.sqrt(a)
        return best_z_errors
"""    

def zz_line_scan(wave,flux,ivar,resolution,lines,vdisps,line_ratio_priors=None,zstep=0.001,zmin=0.,zmax=100.,wave_nsig=3.,ntrack=3,recursive=True) :

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
      line_ratio_priors : dictionnary of priors on line ratio
             the key is the index in lines of this_line
             the value is 1D array of size 3 : 
                 array[0] = index of other_line
                 array[1] = min allowed value of ratio flux[this_line]/flux[other_line]
                 array[1] = max allowed value of ratio flux[this_line]/flux[other_line]
             line_ratio_priors can be set for a limited number of lines

     
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
    log.debug("vdisps=%s"%str(vdisps))
    log.debug("zstep=%f"%zstep)
    log.debug("nframes=%d"%nframes)
    log.debug("nlines=%d"%(lines.size))
    
    # consider Gaussian line profile
    # sig=(line*v/c)*(1+z)
    cspeed=2.9970e5 # km/s
    restframe_sigmas=lines*vdisps/cspeed # sigma in rest-frame in A
    
    # find group of lines that we have to fit together because they overlap
    groups = find_groups(lines,restframe_sigmas)

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
    
    # compute chi2 for zero lines
    chi2_0 = 0.
    for frame_index in range(nframes) :
        chi2_0 += np.sum(ivar[frame_index]*flux[frame_index]**2)
    
    # redshift scan
    best_chi2 = 0
    for z in np.linspace(zmin,zmax,num=int((zmax-zmin)/zstep+1)) :

        # the whole fit happens here
        dchi2,line_amplitudes,line_amplitudes_ivar = zz_line_fit(wave,flux,ivar,resolution,lines,restframe_sigmas,line_ratio_priors,z,wave_nsig,x,gx,groups)
        chi2 = chi2_0 + dchi2
        
        # keep best
        if chi2<best_chi2 or best_chi2==0 :
            best_z_line_amplitudes=line_amplitudes
            best_z_line_amplitudes_ivar=line_amplitudes_ivar

        # now we have to keep track of several solutions
        tracker.add(z=z,chi2=chi2)
    
    log.debug("find_best in range %f %f (nz=%d)"%(tracker.zscan[0],tracker.zscan[-1],len(tracker.zscan)))
    best_zs,best_z_errors,best_chi2s=tracker.find_best(ntrack=ntrack,min_delta_z=0.002)
    best_z_errors[best_z_errors<zstep]=zstep
    
    
    #if not recursive :
    #    log.debug("z = %f +- %f (zstep=%f)"%(best_zs[0],best_z_errors[0],zstep)) 
    #    pylab.plot(tracker.zscan,tracker.chi2scan,"o-")
    #    pylab.show()
 
    if recursive :
        log.debug("first pass best z =%f chi2/ndata=%f, second z=%f dchi2=%f, third z=%f dchi2=%f"%(best_zs[0],best_chi2s[0]/ndata,best_zs[1],best_chi2s[1]-best_chi2s[0],best_zs[2],best_chi2s[2]-best_chi2s[0]))
        #pylab.plot(tracker.zscan,tracker.chi2scan)
        #pylab.show()
        #sys.exit(12)
    
    if recursive :
        # if recursive we refit here all of the best chi2s
        best_results=None

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
            tmp_results = zz_line_scan(wave,flux,ivar,resolution,lines,vdisps=vdisps,line_ratio_priors=line_ratio_priors,zstep=zstep,zmin=zmin,zmax=zmax,wave_nsig=5.,recursive=False,ntrack=1)
            
            
            if rank == 0 :
                # this is the best
                best_results=tmp_results
            else :
                # here we replace the best values
                labels=np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
                for i in range(labels.size,ntrack) :
                    labels=np.append(labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
                
                label=labels[rank]
                keys1=best_results.keys()
                for k1 in keys1 :
                    if k1.find("BEST")==0 :
                        k2=k1.replace("BEST",label)
                        best_results[k2] = tmp_results[k1]
        
        # now that we have finished improving the fits,
        # swap results if it turns out that the ranking has been modified by the improved fit     
        chi2=np.zeros((ntrack))
        for i,l in zip(range(ntrack),rank_labels) :
            chi2[i]=best_results[l+"_CHI2PDF"]
        indices=np.argsort(chi2)
        if np.sum(np.abs(indices-range(ntrack)))>0 : # need swap
            swapped_best_results={}
            for i in range(ntrack) :
                new_label=rank_labels[i]
                old_label=rank_labels[indices[i]]
                for k in best_results :
                    if k.find(old_label)==0 :
                        swapped_best_results[k.replace(old_label,new_label)]=best_results[k]
            #print "DEBUG : has swapped results"
            #print best_results
            #print swapped_best_results
            best_results=swapped_best_results
        
        """
        # check chi2 scan to increase error bar is other local minimum
        # with delta chi2<1
        dz=np.max(np.abs(full_zscan[full_chi2scan<(best_results["BEST_CHI2"]+1)]-best_results["BEST_Z"]))
        #print "DEBUG dz=",dz
        if dz>best_results["BEST_Z_ERR"] :
            log.warning("increase BEST_Z_ERR %f -> %f because of other local minimum"%(best_results["BEST_Z_ERR"],dz))
            best_results["BEST_Z_ERR"]=dz
        """
        log.info("best z=%f+-%f chi2/ndf=%3.2f dchi2=%3.1f"%(best_results["BEST_Z"],best_results["BEST_Z_ERR"],best_results["BEST_CHI2PDF"],best_results["SECOND_CHI2"]-best_results["BEST_CHI2"]))
        return best_results
    
    # here we are outside of the recursive loop
            
    
    ndf=0
    for index in range(nframes) :
        ndf += np.sum(ivar[index]>0)
    ndf-=(np.sum(best_z_line_amplitudes_ivar>0)+1)
    
    snr=math.sqrt(np.sum(best_z_line_amplitudes**2*best_z_line_amplitudes_ivar))
    

    
    
    res={}
    res["BEST_Z"]=best_zs[0]
    res["BEST_Z_ERR"]=best_z_errors[0]
    res["BEST_CHI2"]=best_chi2s[0]
    res["BEST_CHI2PDF"]=best_chi2s[0]/ndf
    res["BEST_SNR"]=snr
    
    for line_index in range(lines.size) :
        res["BEST_AMP_%02d"%line_index]=best_z_line_amplitudes[line_index]
        livar=best_z_line_amplitudes_ivar[line_index]
        res["BEST_AMP_ERR_%02d"%line_index]=(livar>0)/math.sqrt(livar+(livar==0))
    

    return res
