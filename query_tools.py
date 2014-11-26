#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Query tools

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import astropy.units as u

from astropy import coordinates
from astroquery.ned import Ned
from scipy.integrate import romberg

# --------------- #
# -- Constants -- #
# --------------- #

_c = 299792.458     # speed of light in km s^-1
_H_0 = 70.            # Hubble constant in km s^-1 Mpc^-1
_d2r = np.pi/180    # conversion factor from degrees to radians
_O_M = 0.3          # default matter density

# --------------------- #
# -- Basic functions -- #
# --------------------- #

def d_a(z,O_M=_O_M,O_L=None,w=-1,H_0=_H_0,v_dip=None,v_cart=None,v_mon=0,
        coords=None,**kwargs):
    """
    Angular diameter distance in Mpc

    Arguments:
    z -- redshift

    Keyword arguments:
    O_M    -- matter density
    O_L    -- Dark Energy density (will be set to (1 - O_M) if None)
    w      -- Dark Energy equation of state parameter (for constant EOS)
    H_0    -- Hubble constant in km s^-1 Mpc^-1
    """
    if O_L is None:
        Flat = True
        O_L = 1 - O_M
        O_K = 0
    else:
        Flat = False
        O_K = 1 - O_M - O_L

    H_rec = lambda x: 1 / np.sqrt(O_M * (1+x)** 3 +
                                  O_K * (1+x) ** 2 +
                                  O_L * (1+x)**(3*(1+w)))

    integral=romberg(H_rec,0,z)

    if O_K == 0:
        result = _c / (1+z) / H_0 * integral
    elif O_K < 0:
        result = (_c / (1+z) / H_0 / np.sqrt(-O_K) *
                  np.sin(np.sqrt(-O_K)*integral))
    else:
        result = (_c / (1+z) / H_0 / np.sqrt(O_K) *
                  np.sinh(np.sqrt(O_K)*integral))
    
    return result

def load_from_files(*filenames,**kwargs):
    """
    Load data from files.
    
    Requires files to contain header with keys for each column.
    Only columns according to kwargs keys are returned. If keys are
    given dtype must be given (use object for strings).

    Can filter for redshifts and will look for redshift key (named 
    'redshift' or starting with 'z'). If none is found or results 
    ambiguous, it must be stated manually.

    Returns list of as many numpy array as keys (+ fileindex array 
    if return_fileindex is True)
    """
    if 'keys' in kwargs.keys() and 'dtype' not in kwargs.keys():
        raise ValueError('Please set stype as well.')
    elif 'keys' in kwargs.keys() and 'dtype' in kwargs.keys():
        if len(kwargs['keys']) != len(kwargs['dtype']):
            raise ValueError('Length of keys and dtype must match.')

    z_range = kwargs.pop('z_range',None)
    z_key = kwargs.pop('z_key',None)
    keys = kwargs.pop('keys',['Name','RA','Dec','z'])
    dtypes = kwargs.pop('dtype',[object,float,float,float])
    case_sensitive = kwargs.pop('case_sensitive',False)
    comments = kwargs.pop('comments','#')
    delimiter = kwargs.pop('delimeter',None)
    return_fileindex = kwargs.pop('return_fileindex',False)

    if kwargs != {}:
        unknown_kw = ' '.join(kwargs.keys())
        raise TypeError('load_from_files got unknown keyword arguments: {}'.format(unknown_kw))

    if not case_sensitive:
        keys = [a.upper() for a in keys]

    if z_range is not None and z_key is None:
        z_keys = [key for key in keys 
                  if key[0].upper() == 'Z' or key.upper() == "REDSHIFT"] 
        if len(z_keys) == 0:
            raise ValueError('Failed to determine z_key, please set kwarg z_key')
        elif len(z_keys) > 1:
            raise ValueError('Ambiguous z_key, please set kwargs z_key manually')
        else:
            z_key = z_keys[0]

    out = None
    fileindex = []

    for k,filename in enumerate(filenames):
        tmp = np.genfromtxt(filename,names=True,comments=comments,dtype=None,
                            case_sensitive=case_sensitive,delimiter=delimiter)
        
        if z_range is None:
            tmp2 = np.zeros((len(tmp),),dtype=zip(keys,dtypes))
            fileindex.extend([k for a in range(len(tmp))])
            for key in keys:
                tmp2[key] = tmp[key]
        else:
            z_filter = (tmp[z_key] >= z_range[0]) & (tmp[z_key] < z_range[1]) 
            tmp2 = np.zeros((np.sum(z_filter),),dtype=zip(keys,dtypes))
            fileindex.extend([k for a in range(np.sum(z_filter))])
            for key in keys:
                tmp2[key] = tmp[key][z_filter]
                    
        if out is None:
            out = tmp2
        else:
            out = np.concatenate((out,tmp2))
            
    if return_fileindex:
        return [out[key] for key in keys] + [np.array(fileindex)]
    else:
        return [out[key] for key in keys]

# --------------------- #
# -- Query functions -- #
# --------------------- #

def query_coords(RA,Dec,z,diam_transverse=1.,diam_z=0.01,types=['GCluster','Supernova']):
    """
    
    """
    co = coordinates.SkyCoord(ra=RA, dec=Dec,unit=(u.deg, u.deg))
    diam = 1. / d_a(z) / _d2r 
    result_table = Ned.query_region(co, radius=diam * u.deg)
    
    return result_table
