from clib.bellhop import *
import numpy as np
import sys
import subprocess


n_file = int(sys.argv[1])    #file number
times = [0, 11]     #chosen times 
for i in range(len(times)) : 
    s = bellhop({'gs': {'datadir': '/home/datawork-lops-osi/jgula/NESED/', 'file':n_file, 'itime':times[i], 
                        'lon': [-66.6,-65.4], 'lat': [36.,36.], 'plot_map': True, 'contour':True}}, zmax = 4900)
    
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
    s.generate_btyfile(file_env = file_name+'.env', bathy=4500.)
    #Arrivals calculations .arr
    subprocess.call(["bellhop.exe", file_name])
                
    ### ONE PROFILE EVERY 50km
    # ranges 0km, 53km and 107km 
    c_3profiles = np.vstack((c[0,:],c[107,:],c[215,:]))
    r_3profiles = np.array([r[0], r[100], r[215]]) 
    s.generate_envfile('gs',file_env = file_name+'_50km.env', Issp=issp, SSP_depth_step=1, c = c_3profiles[0,:])
    s.generate_sspfile('gs', file_env = file_name+'_50km.env', SSP_depth_step=1, r = r_3profiles, c = c_3profiles)
    s.generate_btyfile(file_env = file_name+'_50km.env', bathy=4500.)
    subprocess.call(["bellhop.exe", file_name+'_50km'])

    
    ### ONE PROFILE EVERY 10km
    c_12profiles = np.vstack((c[0,:],c[20,:],c[40,:],c[60,:],c[80,:],c[100,:], c[120,:], c[140,:], \
                              c[160,:], c[180,:], c[200,:], c[215,:]))
    r_12profiles = np.array([r[0], r[20], r[40], r[60], r[80], r[100], r[120], r[140], r[160], \
                             r[180], r[200], r[215]]) 
    s.generate_envfile('gs',file_env = file_name+'_10km.env', Issp=issp, SSP_depth_step=1, c = c_12profiles[0,:])
    s.generate_sspfile('gs', file_env = file_name+'_10km.env', SSP_depth_step=1, r = r_12profiles, c = c_12profiles)
    s.generate_btyfile(file_env = file_name+'_10km.env', bathy=4500.)
    subprocess.call(["bellhop.exe", file_name+'_10km'])
    
    
    
    
    
    
    
    
    
    

