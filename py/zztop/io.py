"""
zztop.io
=================

IO routines for zztop.
"""

import numpy as np
from astropy.io import fits
from desispec.log import get_logger
import sys

def write_zbest(filename, brickname, zztop_results, truth_table_hdu=None):
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
        ('Z',         np.float32),
        ('ZERR',      np.float32),
        ('TYPE',      'S8'),
        ('SUBTYPE',   'S8')
        ]
    for k in zztop_results.dtype.fields :
        dtype.append((k,zztop_results.dtype[k]))
    
    data = np.empty(zztop_results.size, dtype=dtype)
    
    # required zfitter fields
    
    data['BRICKNAME'] = brickname
    data['Z']         = zztop_results["BEST_Z"]
    data['ZERR']      = zztop_results["BEST_Z_ERR"] # need to redo this
    # target ID and ZWARN are already in zztop results
        
    # need to fill TYPE and SUBTYPE

    # add zztop results
    for k in zztop_results.dtype.names :
        data[k]=zztop_results[k]
    

    hdus = fits.HDUList()
    hdus.append(fits.BinTableHDU(data, name='ZBEST', uint=True))
    
    # add truth table if exists for convenience
    # in development
    if truth_table_hdu is not None :
        hdus.append(truth_table_hdu)
    

    hdus.writeto(filename, clobber=True)

