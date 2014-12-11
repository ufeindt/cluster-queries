#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Query tools

Authors: Ulrich Feindt (feindt@physik.hu-berlin.de, unless noted otherwise)
"""

import numpy as np
import astropy.units as u

from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM

from astroquery.ned import Ned
Ned.TIMEOUT = 600 # Increase time out to 10 minutes

# --------------- #
# -- Constants -- #
# --------------- #

_c = 299792.458     # speed of light in km s^-1
_H_0 = 70.          # Hubble constant in km s^-1 Mpc^-1
_d2r = np.pi/180    # conversion factor from degrees to radians
_O_M = 0.3          # default matter density

cosmo = FlatLambdaCDM(H0=_H_0, Om0=_O_M)

# --------------------- #
# -- Basic functions -- #
# --------------------- #

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

def query_coords(ra,dec,z,r_transverse=1.,r_z=0.01,types=['GClstr'],min_ref=10):
    """
    
    """
    if type(types) == str:
        types = [types]

    if type(r_transverse) in [float,np.float16,np.float32,np.float64]:
        r_transverse = r_transverse * u.Mpc
    elif type(r_transverse) != u.quantity.Quantity:
        raise TypeError('r_transverse must be float or astropy quantity')

    co = coordinates.SkyCoord(ra=ra,dec=dec,unit=(u.deg,u.deg))
    
    r_angular = r_transverse * cosmo.arcsec_per_kpc_proper(z)  
    result_table = Ned.query_region(co, radius=r_angular.to(u.arcmin))
    
    if types is None:
        return {None: _filter_z_references(result_table,z,r_z,min_ref)}
    else:
        out = {}
        for obj_type in types:
            result_by_type = result_table[result_table['Type'].data.data == obj_type]
            if len(result_by_type) > 0:
                out[obj_type] = _filter_z_references(result_by_type,z,r_z,min_ref)
            else:
                out[obj_type] = None
        return out

def _filter_z_references(result_table,z,r_z=0.01,min_ref=10):
    """

    """
    z_filter = ((result_table['Redshift'].data.data >= z - r_z)
                & (result_table['Redshift'].data.data < z + r_z))
    ref_filter = result_table['References'].data.data >= min_ref
    
    result_cut = result_table[z_filter & ref_filter]
    
    ref_sort_idx = np.argsort(result_cut['References'].data.data)[::-1]
    result_sorted = result_cut[ref_sort_idx]

    # Get CMB-centric redshifts
    z_cmb = helio2cmb(result_sorted['Redshift'].data.data,
                      result_sorted['RA(deg)'].data.data,
                      result_sorted['DEC(deg)'].data.data)

    # prepare output array
    names = ['Name','z_helio','z_cmb','RA','DEC','Distance (arcmin)','References','Type']
    dtypes = [object,float,float,float,float,float,int,object]
    out_data = [result_sorted['Object Name'].data.data,
                result_sorted['Redshift'].data.data,
                z_cmb,
                result_sorted['RA(deg)'].data.data,
                result_sorted['DEC(deg)'].data.data,
                result_sorted['References'].data.data,
                result_sorted['Type'].data.data]

    out = np.zeros((len(result_cut),),dtype=zip(names,dtypes))
    for name, data in zip(names,out_data):
        out[name] = data

    return out

def helio2cmb(z, ra, dec):
    """
    Convert z_helio to z_cmb for objects at *(ra,dec)* equatorial 
    coordinates (J2000, in degrees).

    Sources:
    - http://ned.ipac.caltech.edu/help/velc_help.html
    """
    v_apex = 371.0
    l_apex = 264.14
    b_apex = 48.26

    l, b = radec2gcs(ra,dec)

    z_cmb = z + v_apex/_c * (np.sin(b*_d2r) * np.sin(b_apex*_d2r) +
                             np.cos(b*_d2r) * np.cos(b_apex*_d2r) *
                             np.cos((l-l_apex)*_d2r))

    return z_cmb

# -------------------------------- #
# ----  FROM THE SNf ToolBox ----- #
# -------------------------------- #

def radec2gcs(ra, dec, deg=True):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Convert *(ra,dec)* equatorial coordinates (J2000, in degrees if
    *deg*) to Galactic Coordinate System coordinates *(lII,bII)* (in
    degrees if *deg*).

    Sources:

    - http://www.dur.ac.uk/physics.astrolab/py_source/conv.py_source
    - Rotation matrix from
      http://www.astro.rug.nl/software/kapteyn/celestialbackground.html

    .. Note:: This routine is only roughly accurate, probably at the
              arcsec level, and therefore not to be used for
              astrometric purposes. For most accurate conversion, use
              dedicated `kapteyn.celestial.sky2sky` routine.

    >>> radec2gal(123.456, 12.3456)
    (210.82842704243518, 23.787110745502183)
    """

    if deg:
        ra  =  ra * _d2r
        dec = dec * _d2r

    rmat = np.array([[-0.054875539396, -0.873437104728, -0.48383499177 ],
                    [ 0.494109453628, -0.444829594298,  0.7469822487  ],
                    [-0.867666135683, -0.198076389613,  0.455983794521]])
    cosd = np.cos(dec)
    v1 = np.array([np.cos(ra)*cosd,
                  np.sin(ra)*cosd,
                  np.sin(dec)])
    v2 = np.dot(rmat, v1)
    x,y,z = v2

    c,l = rec2pol(x,y)
    r,b = rec2pol(c,z)

    assert np.allclose(r,1), "Precision error"

    if deg:
        l /= _d2r
        b /= _d2r

    return l, b

def rec2pol(x,y, deg=False):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Conversion of rectangular *(x,y)* to polar *(r,theta)*
    coordinates
    """

    r = np.hypot(x,y)
    t = np.arctan2(y,x)
    if deg:
        t /= _d2r

    return r,t
