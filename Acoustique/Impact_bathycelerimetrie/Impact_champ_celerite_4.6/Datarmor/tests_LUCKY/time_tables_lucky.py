from clib.bellhop import *
import numpy as np
import sys
import subprocess

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#bellhop_exe = "/home1/datahome/aponte/at/Bellhop/bellhop.exe"
bellhop_exe = "/home1/datawork/mponchar/cognac/bellhop.exe"

n_file = int(sys.argv[1])    #file number
times = [0]     #chosen times 
for i in range(len(times)) : 
    
    s = bellhop({'gs': {'data_files': '/home/datawork-lops-osi/jgula/LUCKY/*his*.nc', \
                        'hgrid_file': '/home/datawork-lops-osi/jgula/LUCKY/lucky_grd.nc', \
                        'vgrid_file': '/home/datawork-lops-osi/jgula/LUCKY/lucky_his.01000.nc',\
                        'file': n_file, 'itime' : times[i], 'lon': [-33., -31.8], \
                        'lat': [36., 36.], 'plot_map': True, 'contour':True}},  zmax = 2300)
    
    #SSP profiles
    r = s.SSP['gs']['s']              # range (m)
    depth = s.SSP['gs']['depth'][:]   # depth (m)
    c = s.SSP['gs']['c']              # celerity (m/s)
    
    # Parameters Bellhop env filr
    issp=0                                # sound speed profile number
    s.params['hypothesis'] = 'QVWT'       # Q for quadrilateral SSP interpolation (SSP2D)
    s.params['file_type'] = 'A'           # arrivals time and amplitude file (.arr)
    s.params['zs'] = 100.                 # source depth
    s.params['ALimites'] = [-89.0, 89.0]  # limit angles
    #
    s.params['NDepth'] = 51               # number of receiver depths
    s.params['zmin'] = 0.                 # minimum depth (m)
    s.params['zmax'] = 500.               # maximum depth (m) (unuseful if NDepth=1)
    #                 
    s.params['NRange'] = 101              # number of receiver range    
    s.params['rmin'] = 0.                 # minimum range (km)
    s.params['rmax'] = 100.               # maximum range (km) (unuseful if NRange=1)            
    #
    s.params['zbox'] = 5000.              # box depth limit 
    s.params['rbox'] = 101.               # box range limit 
    
    #### FULL SIMULATION 
    file_name = str(n_file)+'_time%d'%times[i]
    s.generate_envfile('gs',file_env = file_name+'.env', Issp=issp, SSP_depth_step=1)
    s.generate_sspfile('gs', file_env = file_name+'.env', SSP_depth_step=1)
    s.generate_btyfile(file_env = file_name+'.env', bathy=4000.)
    #Arrivals calculations .arr
    subprocess.call([bellhop_exe, file_name])
 
    ### ONE PROFILE EVERY 50km
    # ranges 0km, 54km and 108km 
    c_3profiles = np.vstack((c[0,:],c[27,:],c[54,:]))
    r_3profiles = np.array([r[0], r[27], r[54]]) 
    s.generate_envfile('gs',file_env = file_name+'_50km.env', Issp=issp, SSP_depth_step=1, \
                       c=c_3profiles[0,:])
    s.generate_sspfile('gs', file_env = file_name+'_50km.env', SSP_depth_step=1, r = r_3profiles, \
                       c = c_3profiles)
    s.generate_btyfile(file_env = file_name+'_50km.env', bathy=4000.)
    subprocess.call([bellhop_exe, file_name+'_50km'])

    ### ONE PROFILE EVERY 10km
    c_12profiles = np.vstack((c[0,:],c[5,:],c[10,:],c[15,:],c[20,:],c[25,:], c[30,:], c[35,:], \
                              c[40,:], c[45,:], c[50,:], c[54,:]))
    r_12profiles = np.array([r[0], r[5], r[10], r[15], r[20], r[25], r[30], r[35], r[40], \
                             r[45], r[50], r[54]]) 
    s.generate_envfile('gs',file_env = file_name+'_10km.env', Issp=issp, SSP_depth_step=1, \
                       c = c_12profiles[0,:])
    s.generate_sspfile('gs', file_env = file_name+'_10km.env', SSP_depth_step=1, r = r_12profiles, \
                       c = c_12profiles)
    s.generate_btyfile(file_env = file_name+'_10km.env', bathy=4000.)
    subprocess.call([bellhop_exe, file_name+'_10km'])
    
    
    
    ### ONE PROFILE EVERY 6km
    c_19profiles = np.vstack((c[0,:], c[3,:], c[6,:], c[9,:], c[12,:], c[15,:], c[18,:], c[21,:], \
                              c[24,:], c[27,:], c[30,:], c[33,:], c[36,:], c[39,:], c[42,:], \
                              c[45,:], c[48,:], c[51,:], c[54,:]))
    
    r_19profiles = np.array([r[0], r[3], r[6], r[9], r[12], r[15], r[18], r[21], r[24], r[27], \
                             r[30], r[33], r[36], r[39], r[42], r[45], r[48], r[51], r[54]])
    s.generate_envfile('gs',file_env = file_name+'_5km.env', Issp=issp, SSP_depth_step=1, \
                       c = c_19profiles[0,:])
    s.generate_sspfile('gs', file_env = file_name+'_5km.env', SSP_depth_step=1, r = r_19profiles,\
                       c = c_19profiles)
    s.generate_btyfile(file_env = file_name+'_5km.env', bathy=4000.)
    subprocess.call([bellhop_exe, file_name+'_5km'])
    
    
    
    
    
    
    
    

