from desispec.log import get_logger
import numpy as np
import math

def zz_line_scan(wave,flux,ivar,resolution,lines,vdisps,line_ratio_priors=None,zstep=0.001,zmin=0.,zmax=100.,wave_nsig=2.,ntrack=3,recursive=True) :

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
    

    ndf=0
    for index in range(nframes) :
        ndf += np.sum(ivar[index]>0)
    
    ndf-=2 # FOR THE MOMENT WE FIT ONLY ONE AMPLITUDE PER Z
    if ndf<=0 :
        log.warning("ndf=%d skip this spectrum")
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
    restframe_sigmas=lines*vdisps/cspeed # sigma in rest-frame
    

    # find group of lines that we have to fit together because they overlap
    line_group=-1*np.ones((lines.size)).astype(int)
    line_group[0]=0 # add first one

    grouping_debug=False
    
    # grouping lines that overlap
    # add simple resolution estimate for grouping
    resolution_sigma=1. 
    for line_index in range(lines.size) :
        if grouping_debug : log.debug("line_index = %d"%line_index)
        line=lines[line_index]
        if grouping_debug : log.debug('line: %s, line_group: %s'%(line,line_group))
        for group_index in np.where(line_group>=0)[0] :
            if grouping_debug : log.debug('group_index: %d, np.where(line_group>=0)[0]: %s'%(group_index,np.where(line_group>=0)[0]))
            for other_line in lines[np.where(line_group==group_index)[0]] :
                if grouping_debug : log.debug('other_line: %s, np.where(line_group==group_index)[0]: %s, lines[np.where(line_group==group_index)[0]]: %s'%(other_line,np.where(line_group==group_index)[0],lines[np.where(line_group==group_index)[0]]))
                if grouping_debug : log.debug('line-other_line = %s'%(line-other_line))
                if grouping_debug : log.debug('%f <? %f'%((line-other_line)**2,5**2*(restframe_sigmas[line_index]**2+resolution_sigma**2)))
                if (line-other_line)**2 < 5**2*(restframe_sigmas[line_index]**2+resolution_sigma**2) :
                    line_group[line_index]=group_index
                    if grouping_debug : log.debug('line_group: %s'%(line_group))
                    if grouping_debug : log.debug('End of cond. 1')
                    break
            if line_group[line_index]>=0 :
                if grouping_debug : log.debug('line_group: %s'%line_group)
                if grouping_debug : log.debug('End of cond. 2')
                break
        if line_group[line_index]==-1 :
            if grouping_debug : log.debug('line_group[line_index]=-1, lin_index = %d'%line_index)
            line_group[line_index]=np.max(line_group)+1
            if grouping_debug : log.debug('New line_group[line_index]: %s'%line_group[line_index])
    for line,group_index in zip(lines,line_group) :
        if grouping_debug : log.debug("line=%f group=%d"%(line,group_index))
    
    groups={}
    for g in np.unique(line_group) :
        groups[g]=np.where(line_group==g)[0]
    
    for g in groups :
        log.debug("group=%d lines=%s"%(g,groups[g]))
     
    nframes=len(wave)
    zrange=np.zeros((nframes,2))
    
    
    log.debug("ndf=%d"%ndf)
    
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
    
    log.debug("z min=%f z max=%f"%(zmin,zmax))
    
    
    # compute one highres gaussian to go faster and not call exp() after
    x=np.linspace(-5.,5.,100)
    gx=1./math.sqrt(2*math.pi)*np.exp(-0.5*x**2)
    
    nbz=ntrack
    best_zs=np.zeros((nbz))
    best_z_errors=np.zeros((nbz))
    best_chi2s=1e12*np.ones((nbz))
    chi2s_at_best_z_minus_zstep=np.zeros((nbz))
    chi2s_at_best_z_plus_zstep=np.zeros((nbz))
    
    previous_chi2=0
    previous_is_at_rank=-1
    
    model=[]
    for frame_index in range(nframes) :
#        model.append(np.zeros((wave[index].size)))
        model.append(np.zeros((wave[frame_index].size))) # was it a bug ? (CB)
        


    # redshift scan (we just record here the best value)
    
    line_amplitudes=np.zeros((lines.size))              # used to compute chi2 after line amplitude fit
    line_amplitudes_ivar=np.zeros((lines.size))              # used to compute chi2 after line amplitude fit
    sum_ivar_flux_prof = np.zeros((nframes,lines.size)) # used to compute chi2 after line amplitude fit
    sum_ivar_prof2     = np.zeros((nframes,lines.size)) # used to compute chi2 after line amplitude fit

    best_z_line_amplitudes=np.zeros((lines.size))         # save this
    best_z_line_amplitudes_ivar=np.zeros((lines.size))  # save this
    
    # compute chi2 for zero lines
    chi2_0 = 0.
    for frame_index in range(nframes) :
        chi2_0 += np.sum(ivar[frame_index]*flux[frame_index]**2)
    

    for z in np.linspace(zmin,zmax,num=int((zmax-zmin)/zstep+1)) :
        
        # reset (wo memory allocation)
        line_amplitudes *= 0.
        for frame_index in range(nframes) :
            model[frame_index] *= 0.
        sum_ivar_flux_prof *= 0;
        sum_ivar_prof2     *= 0;
        chi2 = chi2_0
        
        redshifted_lines=lines*(1+z)
        redshifted_sigmas=restframe_sigmas*(1+z) # this is the sigmas of all lines
        
        delta_chi2_coeff_per_group={} # either a scalar or a vector per group (it's getting complicated)
            
        # compute profiles for each line in each group and fill model
        for group_index in groups :
            group=groups[group_index]
            l1=np.min(redshifted_lines[group]-wave_nsig*redshifted_sigmas[group])
            l2=np.max(redshifted_lines[group]+wave_nsig*redshifted_sigmas[group])
            
            nlines=group.size

            # matrices of system to solve :   a*line_amplitudes = b
            if nlines==1 :
                a0=0.
                b0=0.
            else :
                a=np.zeros((nlines,nlines))
                b=np.zeros((nlines))
            
            for frame_index in range(nframes) :
                frame_wave = wave[frame_index]
                frame_ivar = ivar[frame_index]
                w1=np.min(frame_wave[frame_ivar>0])
                w2=np.max(frame_wave[frame_ivar>0])
                if w2<l1 :
                    continue
                if w1>l2 :
                    continue
                # wavelength that matter :
                wave_index=np.where((frame_wave>=l1)&(frame_wave<=l2)&(frame_ivar>0))[0]
                if wave_index.size == 0 :
                    continue
                frame_wave=frame_wave[wave_index]
                frame_ivar=frame_ivar[wave_index]
                frame_flux=flux[frame_index][wave_index]
                
                # compute profiles :
                tmp_prof=np.zeros((nlines,wave_index.size))
                 
                for i,line_index,line,sig in zip(range(group.size),group,redshifted_lines[group],redshifted_sigmas[group]) :
                    prof=np.interp((frame_wave-line)/sig,x,gx)
                    # save stuff
                    sum_ivar_flux_prof[frame_index,line_index]=np.sum(frame_ivar*prof*frame_flux)
                    sum_ivar_prof2[frame_index,line_index]=np.sum(frame_ivar*prof**2)
                    tmp_prof[i]=prof

                # fill amplitude system (a and b) :
                if nlines==1 :
                    a0+=sum_ivar_prof2[frame_index,group[0]]
                    b0+=sum_ivar_flux_prof[frame_index,group[0]]
                else :
                    for i in range(nlines) :
                        b[i]   += sum_ivar_flux_prof[frame_index,group[i]]
                        for j in range(nlines) :
                            a[i,j] += np.sum(frame_ivar*tmp_prof[i]*tmp_prof[j])
                    
            # now solve
            if nlines==1 :
                if a0>0 :
                    line_amplitudes[group[0]] = max(0.,b0/a0)
                    line_amplitudes_ivar[group[0]] = a0
                else :
                    line_amplitudes[group[0]] = 0
                    line_amplitudes_ivar[group[0]] = 0 
            else :
                for i in range(nlines) :
                    line_amplitudes_ivar[group[i]]=a[i,i]
                    a[i,i] += (a[i,i]==0)
                    
                try :
                    amps=desispec.linalg.cholesky_solve(a,b)
                    for i,line_index in zip(range(nlines),group) :
                        line_amplitudes[line_index]=max(0.,amps[i])
                except  :
                    for i,line_index in zip(range(nlines),group) :
                        line_amplitudes[line_index]=0.

            # add chi2 for this group
            if nlines==1 :
                if a0>0 :
                    delta_chi2_coeff_per_group[group_index] = -b0
                    #chi2 -= b0**2/a0 # don't do it now because prior can modify line_amplitudes
                else :
                    delta_chi2_coeff_per_group[group_index] = 0.
                    
            else :
                delta_chi2_coeff_per_group[group_index] = -b
                #chi2 -= np.inner(b,line_amplitudes[group]) # don't do it now because prior can modify line_amplitudes
                
        

        # apply priors 
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


        # apply delta chi2 per group
        for group_index in groups :
            group=groups[group_index]
            if group.size==1 :
                chi2 += delta_chi2_coeff_per_group[group_index]*line_amplitudes[group[0]]
            else :
                chi2 += np.inner(delta_chi2_coeff_per_group[group_index],line_amplitudes[group])

        if chi2<np.max(best_chi2s) :
            
            # find position depending on value of chi2
            i=np.where(chi2<best_chi2s)[0][0]
            # do we replace or insert and shift the other
            if abs(z-best_zs[i])>0.0033*(1+z) : # outside of catas. failure range
                # we insert, meaning we have to shift the others
                # else we replace and have nothing to do here
                best_zs[i+1:]=best_zs[i:-1]
                best_chi2s[i+1:]=best_chi2s[i:-1]
                chi2s_at_best_z_minus_zstep[i+1:]=chi2s_at_best_z_minus_zstep[i:-1]
                chi2s_at_best_z_plus_zstep[i+1:]=chi2s_at_best_z_plus_zstep[i:-1]
            
            best_chi2s[i]=chi2
            best_zs[i]=z
            chi2s_at_best_z_minus_zstep[i]=previous_chi2
            best_z_line_amplitudes=line_amplitudes
            best_z_line_amplitudes_ivar=np.sum(sum_ivar_prof2,axis=0)            
            previous_is_at_rank=i
        else :
            if previous_is_at_rank>=0 :
                chi2s_at_best_z_plus_zstep[previous_is_at_rank]=chi2
            previous_is_at_rank=-1

        #log.debug("scan z =%f chi2/ndf=%f"%(z,chi2/ndf))

        previous_chi2=chi2
        
    
    
    
    if recursive :
        log.debug("first pass best z =%f chi2/ndf=%f, second z=%f dchi2=%f, third z=%f dchi2=%f"%(best_zs[0],best_chi2s[0]/ndf,best_zs[1],best_chi2s[1]-best_chi2s[0],best_zs[2],best_chi2s[2]-best_chi2s[0]))
        
    
    
    for rank in range(best_zs.size) :
        # we can use the values about best_chi2 to guess the uncertainty on z with a polynomial fit
        coeffs=np.polyfit([best_zs[rank]-zstep,best_zs[rank],best_zs[rank]+zstep],[chi2s_at_best_z_minus_zstep[rank],best_chi2s[rank],chi2s_at_best_z_plus_zstep[rank]],2)
        a=coeffs[0]
        b=coeffs[1]
        c=coeffs[2]
    
        best_z_errors[rank] = zstep
        if a>0 :
            best_zs[rank]       = -b/(2*a)
            best_chi2s[rank]    = c-b**2/(4*a)
            best_z_errors[rank] = 1./math.sqrt(a)
    
    
    if recursive :
        best_results=None
        
        rank_labels = np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(rank_labels.size,ntrack) :
            rank_labels=np.append(rank_labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
        
        for rank in range(best_zs.size) :
            # second loop about minimum
            # where we save things to compute errors
            tmp_z       = best_zs[rank]
            tmp_z_error = best_z_errors[rank]
            if tmp_z_error>zstep :
                tmp_z_error = zstep
            tmp_z_error=max(tmp_z_error,0.0001)
            z_nsig=2.
            zmin=tmp_z-z_nsig*tmp_z_error
            zmax=tmp_z+z_nsig*tmp_z_error
            zstep=(zmax-zmin)/10
            tmp_results = zz_line_scan(wave,flux,ivar,resolution,lines,vdisps=vdisps,line_ratio_priors=line_ratio_priors,zstep=zstep,zmin=zmin,zmax=zmax,wave_nsig=5.,recursive=False)

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
        
        # swap results if it turns out that the ranking has been modified by the improved fit     
        chi2=np.zeros((ntrack))
        for i,l in zip(range(ntrack),rank_labels) :
            chi2[i]=best_results[l+"_CHI2"]
        indices=np.argsort(chi2)
        if np.sum(np.abs(indices-range(ntrack)))>0 : # need swap
            swapped_best_results={}
            for i in range(ntrack) :
                new_label=rank_labels[i]
                old_label=rank_labels[indices[i]]
                for k in best_results :
                    if k.find(old_label)==0 :
                        swapped_best_results[k.replace(old_label,new_label)]=best_results[k]
            best_results=swapped_best_results

        log.info("best z=%f+-%f chi2/ndf=%3.2f snr=%3.1f dchi2=%3.1f"%(best_results["BEST_Z"],best_results["BEST_Z_ERR"],best_results["BEST_CHI2PDF"],best_results["BEST_SNR"],best_results["SECOND_CHI2"]-best_results["BEST_CHI2"]))
        return best_results
    
    
            
    
    ndf=0
    for index in range(nframes) :
        ndf += np.sum(ivar[index]>0)
    
    ndf-=(np.sum(best_z_line_amplitudes>0)+1)
    
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
    

    """
    # plotting
    for line_index in range(lines.size) :
        print "line %fA amp= %g +- %g\n"%(lines[line_index],res["BEST_AMP_%02d"%line_index],res["BEST_AMP_ERR_%02d"%line_index])
    
    best_z = best_zs[0]
    redshifted_lines=lines*(1+best_z)
    redshifted_sigmas=restframe_sigmas*(1+best_z) # this is the sigmas of all lines
    colors=np.array(["b","g","r"])
    for frame_index in range(nframes) :
        color=colors[frame_index%3]
        ok=np.where(ivar[frame_index]>1./(1.e-17)**2)[0]
#        pylab.errorbar(wave[frame_index][ok],flux[frame_index][ok],1/np.math.sqrt(ivar[frame_index][ok]),fmt="o",color=color,alpha=0.3)
        model=0.*flux[frame_index]
        for line,sig,amp in zip(redshifted_lines,redshifted_sigmas,best_z_line_amplitudes) :
            if amp>0 :
                model += amp*np.interp((wave[frame_index]-line)/sig,x,gx)

        for line in redshifted_lines:
            ok=np.where((wave[frame_index]>line-10.) & (wave[frame_index]<line+10.))
#            pylab.plot(wave[frame_index][ok]/(1.+best_z),model[ok],c="r")
#            pylab.plot(wave[frame_index][ok]/(1.+best_z),flux[frame_index][ok])

#    pylab.show()
    """    

    return res
