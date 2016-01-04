"""
zztop.io
=================

IO routines for zztop.
"""

import numpy as np
from astropy.io import fits
from desispec.log import get_logger
import sys

def write_zbest(filename, brickname, targetids, zztop_results, truth_table_hdu=None):
    """Writes zztop output to ``filename``.

    Args:
        filename : full path to output file
        brickname : brick name, e.g. '1234p5678'
        targetids[nspec] : 1D array of target IDs
        zztop_results[nspec] : zztop results
        truth_table : truth if exists

    """
    dtype = [
        ('BRICKNAME', 'S8'),
        ('TARGETID',  np.int64),
        ('Z',         np.float32),
        ('ZERR',      np.float32),
        ('ZWARN',     np.int64),
        ('TYPE',      'S8'),
        ('SUBTYPE',   'S8')
        ]
    for k in zztop_results.dtype.fields :
        dtype.append((k,zztop_results.dtype[k]))
    
    data = np.empty(zztop_results.size, dtype=dtype)
    
    data['BRICKNAME'] = brickname
    data['TARGETID']  = targetids

    # change things here
    print zztop_results.dtype.names
    print zztop_results.size
    
    
    
    data['Z']         = zztop_results["BEST_Z"]
    data['ZERR']      = zztop_results["BEST_Z_ERR"]
    # need to rafine this
    data['ZWARN']     = zztop_results["BEST_CHI2"]<zztop_results["SECOND_CHI2"]-9.
    # need to fill TYPE and SUBTYPE
    for k in zztop_results.dtype.names :
        data[k]=zztop_results[k]
    

    hdus = fits.HDUList()
    hdus.append(fits.BinTableHDU(data, name='ZBEST', uint=True))
    if truth_table_hdu is not None :
        hdus.append(truth_table_hdu)
    

    hdus.writeto(filename, clobber=True)

