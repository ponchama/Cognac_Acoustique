# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:25:44 2018

@author: marieponchart
"""
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
#import os
#import pickle
#import numpy as np 
#import xarray as xr
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import scipy.io as sio
#from scipy import interpolate
#import mpl_toolkits.mplot3d.art3d as art3d

#from glob import glob

#from .flow import *



def plotray (filename, dist = False, colors=None, zoom=False):
    ''' 
    Parameters
    ----------
    filename : str
        Output file from bellhop simulation (.ray) 
    '''
        
    file = Path("./%s" %filename)
    if not file.is_file():
        print("Le fichier %s n'exite pas dans ce répertoire." %filename)
        return

    Nsxyz       = np.zeros(3) 
    NBeamAngles = np.zeros(2)

    fid = open(filename,'r')
    title = fid.readline()
    freq  = float( fid.readline() )
    theline = str( fid.readline() )
    datai = theline.split()
    Nsxyz[0] = int( datai[0] )
    Nsxyz[1] = int( datai[1] )
    Nsxyz[2] = int( datai[2] )
    theline = str( fid.readline() )
    datai = theline.split()
    NBeamAngles[0] = int( datai[0] )
    NBeamAngles[1] = int( datai[1] )
    DEPTHT = float( fid.readline() )
    DEPTHB = float( fid.readline() )
    Type   = fid.readline()
    Nsx = int( Nsxyz[0] )
    Nsy = int( Nsxyz[1] )
    Nsz = int( Nsxyz[2] )
    Nalpha = int( NBeamAngles[0] )
    Nbeta  = int( NBeamAngles[1] )
    # axis limits
    rmin =  1.0e9
    rmax = -1.0e9
    zmin =  1.0e9
    zmax = -1.0e9

    #plt.figure(figsize=(9,6))

    for isz in range(Nsz):
        for ibeam in range(Nalpha):
            theline = str( fid.readline() )
            l = len( theline )
            if l > 0:
               alpha0 = float( theline )
               theline = str( fid.readline() )
               datai = theline.split()
               nsteps    = int( datai[0] )
               NumTopBnc = int( datai[1] )
               NumBotBnc = int( datai[2] )
               r = np.zeros(nsteps)
               z = np.zeros(nsteps)
               for j in range(nsteps):
                   theline = str(fid.readline())
                   rz = theline.split()
                   r[j] = float( rz[0] )
                   z[j] = float( rz[1] )        
               rmin = min( [ min(r), rmin ] )
               rmax = max( [ max(r), rmax ] )
               zmin = min( [ min(z), zmin ] )
               zmax = max( [ max(z), zmax ] )

               ## traveled distance 
               d = 0
               for i in range (len(r)-1):
                   d += np.sqrt((r[i+1]-r[i])**2 + (z[i+1]-z[i])**2) 
                    
               ## Color of the ray
              
               if colors is None : 
                   #RED : no reflexion on top and bottom
                   if np.logical_and (NumTopBnc==0, NumBotBnc==0):
                        color = 'r'
                   #BLUE : no reflexion on bottom
                   elif NumBotBnc==0 :
                        color = 'b'
                   #BLACK : reflexion on top and bottom
                   else : 
                        color = 'k'

               else : 
                   color = colors
                    
                    
               ## plot  
               if dist :
                    label = '%.2fm' %d
               else : 
                    label = None
               plt.plot( r/1000., -z,  color = color, label = label)
               plt.axis([rmin/1000.,rmax/1000.,-zmax,-zmin])
    
    plt.title(filename)
    plt.xlabel('range (km)')
    plt.ylabel('profondeur (m)')
    plt.grid()
    
    if zoom :
        plt.ylim ([-500, 0])
    
    if dist : 
        plt.legend()
    plt.savefig('plotray.png', dpi=100)

    fid.close()
    
plotray('gs_1prof_R.ray')
