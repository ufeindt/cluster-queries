#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example NED query of for galaxy clusters at the SN coordinates 

Authors: Ulrich Feindt (ulrich.feindt@fysik.su.se, unless noted otherwise)
"""

import query_tools as qt


if __name__ == '__main__':
    # Names, RAs, Decs, zs = qt.load_from_files('data/snf_caballo_full.dat')
    Names, RAs, Decs, zs = qt.load_from_files('test.dat')
    
    for Name, RA, Dec, z in zip(Names,RAs,Decs,zs):
        try:
            print Name
            query = qt.query_coords(RA,Dec,z)
            if query['GClstr'] is not None and len(query['GClstr']) > 0:
                print query['GClstr']
            else:
                print "nothing found"
        except:
            # If something goes wrong it is probably a timeout
            print "Something went wrong"
        print
