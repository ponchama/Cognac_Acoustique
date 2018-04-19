from pathlib import Path
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.io as sio

from glob import glob

from .flow import *

class bellhop(object):
    ''' bellhop class, contains simulations parameters, generates input files,
    read output files and plot results
    '''
    
    
    def __init__(self, SSP, **kwargs):
        '''
        Parameters
        ----------
        SSP : dict
            Sound speed profile key and file 
            (example : {'p1': './SSP_4profils.mat'})
        '''
    
        zmax = 2550.
        rmax = 100.
        simu = 'simulation'
        self.params = {'name': simu, 'freq': 3000., \
                       'zs': 50., 'zmax': zmax, \
                       'rmax': rmax, 'NDepth': zmax + 1., \
                       'NRange': rmax * 100. + 1., \
                       'ALimites': [-15., 15.], 'file_type': 'R'}
       
        self.params.update(kwargs)
        self.params.update(NDepth = self.params['zmax'] + 1.)
        self.params.update(NRange = self.params['rmax']*100 + 1.)
        
        self.params['file_bathy'] = self.params['name']+'.bty'
        self.params['file_env'] = self.params['name']+'.env'
        self.params['file_ssp'] = self.params['name']+'.ssp'   # range dependent SSP
        #
        self.SSP = {}
        self.load_SSP(SSP)
        

        
        
    def load_SSP(self, SSP_dict):
        '''
        Parameters
        ----------
        SSP_dict : dict
            Sound speed profile dictionnary whose keys represent profile names and items 
            are either a .mat file or a dict containing parameters in order to create 
            profiles from a numerical simulation
        '''
        
        for ssp_key, item in SSP_dict.items():
            if '.mat' in item:
                D = sio.loadmat(item)
                self.SSP[ssp_key] = {'c': D['SSP_Dr1m'], 'depth': D['Depthr'][0,:]}
            elif isinstance(item, dict):
                c, depth = self.compute_SSP_from_flow(**item)
                self.SSP[ssp_key] = {'c': c[::-1].T, 'depth': depth[::-1][:,0]} 
                
              
            
            
    def compute_SSP_from_flow(self, file=None, datadir=None, hgrid_file=None, \
                              lon=None, lat=None, i_eta=None, i_xi=None, L=None, \
                              itime=0, plot_map=False, contour=False, **kwargs):
        # load grid info
        grd = grid(datadir=datadir, hgrid_file=hgrid_file)
        
        # output file
        ofiles = sorted(glob(grd._datadir+'*.*.nc'))
        if file is None:
            file = ofiles[0]
            print('Uses the following output file: %s' %file)
    
        # load T/S
        #T = Dataset(file)['temp']
        #S = Dataset(file)['salt']
        ds = xr.open_dataset(file, chunks = {'s_rho': 1}).isel(time=itime)
        #ds['lon_rho'] = (('eta_rho', 'xi_rho'), grd.lon_rho)
        #ds['lat_rho'] = (('eta_rho', 'xi_rho'), grd.lat_rho)
        
        if lon is not None and lat is not None:
            # xarray way, too slow at the moment
            #d = (lon-ds['lon_rho'])**2 + (lat-ds['lat_rho'])**2
            #T = ds['temp'].where(d == d.min(), drop=True)
            #S = ds['salt'].where(d == d.min(), drop=True)
            # numpy way, fast
            d = (lon-grd.lon_rho)**2 + (lat-grd.lat_rho)**2
            i_eta, i_xi = np.unravel_index(d.argmin(), d.shape)
            T = ds['temp'].isel(eta_rho = i_eta, xi_rho = i_xi)
            S = ds['salt'].isel(eta_rho = i_eta, xi_rho = i_xi)
        
        if plot_map:
            
            crs = ccrs.PlateCarree()

            fig=plt.figure(figsize=(15,10))
            #
            ax=plt.axes(projection=crs)
            ax.set_extent(grd.hextent, crs)
            gl=ax.gridlines(crs=crs,draw_labels=True)
            gl.xlabels_top = False
            ax.coastlines(resolution='50m')
            #
            toplt = ds['temp'].isel(s_rho=-1).values
            # should probably mask T
            cmap = plt.get_cmap('magma')
            im = ax.pcolormesh(grd.lon_rho,grd.lat_rho,toplt,
                               vmin=toplt.min(),vmax=toplt.max(), 
                               cmap=cmap)
            cbar = plt.colorbar(im, format='%.1f', extend='both')
            plt.plot(lon, lat, '*',
                     markeredgecolor='white', markerfacecolor='cadetblue', markersize=20)
            ax.set_title('surface temperature [degC]')
           
        if contour :
            
            cp = plt.contour(grd.lon_rho, grd.lat_rho, grd.h,[500,1000,2500],colors='white')
            plt.clabel(cp, inline=1, fmt='%d', fontsize=10)
        
        
        h = grd.h[None,i_eta,i_xi]
        zeta = ds['zeta'].isel(eta_rho = i_eta, xi_rho = i_xi).values
        z = grd.get_z(zeta, h, grd.sc_r[:,None], grd.Cs_r[:,None])
    
        # build a uniform grid
        z_uni = z[:,[0]] # output z
        #T_uni = interp2z0(z_uni, z, T.values)
        #S_uni = interp2z0(z_uni, z, S.values)
        T_uni = interp2z_1d(z_uni[:,[0]], z[:,0], T.values)
        S_uni = interp2z_1d(z_uni[:,[0]], z[:,0], S.values)
        #lat = grd.lat_rho[i_eta,i_xi]
        c = get_soundc(T_uni, S_uni, z_uni, lon, lat)

        # tmp
        #self.grd = grd        
        #self.ds = ds
        #self.d = d
        #self.dmin = dmin
        #self.T = T
        #self.S = S
        #self.c = c
        
        return c, -z_uni
    
          
        
        
    def generate_envfile(self, ssp_key, Issp=0, file_env=None, SSP_depth_step=1):
        ''' 
        Parameters
        ----------
        ssp_key: str
            Key referencing which sound speed profile should be used
        Issp : int
            Sound speed profile index in SSP file
        file_env : str
            Name of the env file to generate (.env)
        '''
        
       
        # name of the environnement file
        if file_env is None:
            file_env = self.params['file_env']
        print('Output file is : '+file_env)
            
        # load sound speed profile
        c = self.SSP[ssp_key]['c'][Issp,:]
        depth = self.SSP[ssp_key]['depth'][:]

        # Create environment file
        with open(file_env, 'w') as f:
            f.write('\'Range dep, Gaussian beams\'\n')
            f.write('%.1f\n' % self.params['freq'])
            f.write('%.d\n' %1)
            f.write('\'SVWT\'\n')
            #
            depth_max = depth.max() # ! Bathymetrie maximale
            f.write('%d %.1f %.1f\n'%(0, 0.0, depth_max))
            for i in range(0,len(depth),SSP_depth_step):
                f.write('%.1f %.1f /\n' %(depth[i], c[i]))
            #
            f.write('\'A*\' %.1f\n' %0.0)
            f.write('%.1f %.1f %.1f %.1f %.1f %.1f\n'%(depth_max,1800,10,2,0.1,0)) #gravier
            #
            f.write('%d \t  !NSD\n' %1)
            f.write('%d / \t  !Source depth\n' % self.params['zs'])
            f.write('%d \t  !Number receiver depth\n' % self.params['NDepth'])
            f.write('%.1f %.1f / \t  !Receiver depths\n' %(0.0, self.params['zmax']))

            f.write('%d \t  !Number of ranges\n' % self.params['NRange'])
            f.write('%.1f  %.1f / \t  !Range limites\n'%(0., self.params['rmax']))
            f.write('\'%s\'\n' %self.params['file_type'])  # R : .ray ; IB : .shd
            #f.write('\'IB\'\n')            #! On choisit de travailler en champ incoherent
            f.write('%d  \t  !NBeams\n'%0)      #! Bellhop calcule le nombre optimal de rayons
            f.write('%.1f  %.1f  / \t  !Angles limites\n' %( 
                self.params['ALimites'][0],self.params['ALimites'][1]))
            f.write('%.1f  %.1f  %.1f \t  !Steps zbox(m) rbox(km)\n' %(
                0.0,self.params['zmax']+450,self.params['rmax']+1.))

      
            
    def plotray (self, filename = None):
        ''' 
        Parameters
        ----------
        filename : str
            Output file from bellhop simulation (.ray) 
        '''
        
        if filename is None :
            filename = self.params['name']+'.ray'
            
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

                   ## Color of the ray
                   #RED : no reflexion on top and bottom
                   if np.logical_and (NumTopBnc==0, NumBotBnc==0):
                        color = 'r'
                   #BLUE : no reflexion on bottom
                   elif NumBotBnc==0 :
                        color = 'b'
                   #BLACK : reflexion on top and bottom
                   else : 
                        color = 'k'

                   ## plot  
                   plt.plot( r, -z,  color = color )
                   plt.axis([rmin,rmax,-zmax,-zmin])


        plt.title(filename[:-4])
        plt.xlabel('range (m)')
        plt.ylabel('profondeur (m)')
        #plt.savefig('plotray_'+filename[:-4], dpi=100)

        fid.close()

        
        
       
    def readshd(self, filename=None):
        ''' 
        Parameters
        ----------
        filename : str
            Output file from bellhop simulation (.shd)
        '''
        
        if filename is None :            
            filename = self.params['name']+'.shd'
            
        file = Path("./%s" %filename)
        if not file.is_file():
            print("Le fichier %s n'exite pas." %filename)
            return
              
           
        ###### READ file #######  
        
        fid = open(filename,'rb')
        recl  = int( np.fromfile( fid, np.int32, 1 ) )
        title = fid.read(80)
        fid.seek( 4*recl )
        PlotType = fid.read(10)
        fid.seek( 2*4*recl ); # reposition to end of second record
        Nfreq  = int(   np.fromfile( fid, np.int32  , 1 ) )
        Ntheta = int(   np.fromfile( fid, np.int32  , 1 ) )
        Nsx    = int(   np.fromfile( fid, np.int32  , 1 ) )
        Nsy    = int(   np.fromfile( fid, np.int32  , 1 ) )
        Nsd    = int(   np.fromfile( fid, np.int32  , 1 ) )
        Nrd    = int(   np.fromfile( fid, np.int32  , 1 ) )
        Nrr    = int(   np.fromfile( fid, np.int32  , 1 ) ) 
        atten  = float( np.fromfile( fid, np.float32, 1 ) )
        fid.seek( 3 * 4 * recl ); # reposition to end of record 3
        freqVec = np.fromfile( fid, np.float32, Nfreq  )
        fid.seek( 4 * 4 * recl ); # reposition to end of record 4
        thetas  = np.fromfile( fid, np.float32, Ntheta )
        if  ( PlotType[ 0 : 1 ] != 'TL' ):
            fid.seek( 5 * 4 * recl ); # reposition to end of record 4
            Xs     = np.fromfile( fid, np.float32, Nsx )
            fid.seek( 6 * 4 * recl );  # reposition to end of record 5
            Ys     = np.fromfile( fid, np.float32, Nsy )
        else:   # compressed format for TL from FIELD3D
            fid.seek( 5 * 4 * recl ); # reposition to end of record 4
            Pos_S_x     = np.fromfile( fid, np.float32, 2 )
            Xs          = np.linspace( Pos_S_x[0], Pos_S_x[1], Nsx )
            fid.seek( 6 * 4 * recl ); # reposition to end of record 5
            Pos_S_y     = np.fromfile( fid, np.float32, 2 )
            Ys          = np.linspace( Pos_S_y[0], Pos_S_y[1], Nsy )
        fid.seek( 7 * 4 * recl ) # reposition to end of record 6
        zs = np.fromfile( fid, np.float32, Nsd )
        fid.seek( 8 * 4 * recl ) # reposition to end of record 7
        zarray =  np.fromfile( fid, np.float32, Nrd )
        fid.seek( 9 * 4 * recl ) # reposition to end of record 8
        rarray =  np.fromfile( fid, np.float32, Nrr )
        if PlotType == 'rectilin  ':
            pressure = np.zeros( (Ntheta, Nsd, Nrd, Nrr) ) + 1j*np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
            Nrcvrs_per_range = Nrd
        elif PlotType == 'irregular ':
            pressure = np.zeros( (Ntheta, Nsd,   1, Nrr) ) + 1j*np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
            Nrcvrs_per_range = 1
        else:
            pressure = np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
            Nrcvrs_per_range = Nrd
        pressure = np.zeros( (Ntheta,Nsd,Nrcvrs_per_range,Nrr) ) + 1j*np.zeros(    (Ntheta,Nsd,Nrcvrs_per_range,Nrr) )
    
        for itheta in range(Ntheta):
            for isd in range( Nsd ):
                for ird in range( Nrcvrs_per_range ):
                    recnum = 10 + itheta * Nsd * Nrcvrs_per_range + isd * Nrcvrs_per_range + ird
                    status = fid.seek( recnum * 4 * recl ) # Move to end of previous record    
                    if ( status == -1 ):
                        print ('Seek to specified record failed in readshd...')
                    temp = np.fromfile( fid, np.float32, 2 * Nrr ) # Read complex data
                    for k in range(Nrr):
                        pressure[ itheta, isd, ird, k ] = temp[ 2 * k ] + 1j * temp[ 2*k + 1 ]

            fid.close()
            
        geometry = {"zs":zs, "f":freqVec,"thetas":thetas,"rarray":rarray,"zarray":zarray}
        
        return geometry, pressure
            
        
        
        
    def plotshd (self, geometry, pressure): 
        ''' 
        Parameters
        ----------
        geometry : dic
            Output from readshd function. Contains depth and range arrays.
        pressure : array
            output from readshd function. Array of pressure given by bellhop simulation.
        '''
        
        # range and depth
        rt = geometry.get ("rarray")
        zt = geometry.get ("zarray")

        # pressure
        P = np.squeeze(pressure).real
        Pabs = abs(P)
        Pabs[np.where(np.isnan(Pabs))] = 1e-6  #remove NaNs
        Pabs[np.where(np.isinf(Pabs))] = 1e-6  #remove infinities
        Pabs[np.where(Pabs<1e-37)] = 1e-37     #remove zeros

        # transmission loss
        TL = -20.0 * np.log10 (Pabs) 

        #statistics to define limits of colorbar
        icount = TL[np.where(Pabs > 1e-37)]       # statistics only on pressure P > 1e-37
        tlmed = np.median (icount)                # median value
        tlstd = np.std(icount)                    # standard deviation
        tlmax = tlmed + 0.75 * tlstd              # max for colorbar
        tlmax = 10 * np.round (tlmax/10)          # make sure the limits are round numbers
        tlmin = tlmax - 50                        # min for colorbar


        # plot TL from 0 to 500m (ZOOM) 
        plt.pcolormesh (rt, zt, TL, cmap='jet')
        plt.title ('TL - ZOOM de 0 à 500m')
        plt.xlabel("range (m)")
        plt.ylabel("depth (m)")
        cbar = plt.colorbar()
        cbar.set_label("TL(dB)")
        plt.clim ([tlmin,tlmax])
        plt.ylim(ymax = 500)
        plt.gca().invert_yaxis()
        #plt.savefig('plotshd_'+filename[:-4]+'_ZOOM', dpi=100)
        
        
        
        
              
    def plot_all(self, file_ray =None, file_shd=None):
        ''' 
        Parameters
        ----------
        file_ray : str
            Output file from bellhop simulation (.ray)
        file_shd : str
            Output file from bellhop simulation (.shd)
        '''
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        self.plotray(filename = file_ray)
        #
        geometry, pressure = self.readshd(filename = file_shd)
        plt.subplot(1,2,2)
        self.plotshd (geometry, pressure)
  
        
    
    def plotssp (self,ssp_key, Issp=0, y_zoom =150):
        ''' 
        Parameters
        ----------
        ssp_key: str
            Key referencing which sound speed profile should be used
        Issp : int
            Sound speed profile index in SSP file
        '''
        c = self.SSP[ssp_key]['c'][Issp,:]
        depth = self.SSP[ssp_key]['depth'][:]

        plt.figure(figsize=(10,3))
        
        plt.subplot(1,2,1)
        plt.plot(c, depth)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.title('celerity profile : '+self.params['name']+'_SSP%.d' %(Issp+1))
        plt.xlabel('celerity (m/s)')
        plt.ylabel('depth (m)')
    
        plt.subplot(1,2,2)
        plt.plot(c, depth)
        plt.ylim(-10, y_zoom)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.title('Zoom')
        plt.xlabel('celerity (m/s)')
        plt.ylabel('depth (m)')
        
        

    
    
    def plotssp2D (self, filename=None):
        '''
        Parameters
        ----------
        filename: str
            File containing range dependent sound speed profiles (.ssp)    
        '''
        
        if filename is None :           
            filename = self.params['file_ssp']
            
        file = Path("./%s" %filename)
        if not file.is_file():
            print("Le fichier %s n'exite pas dans ce répertoire." %filename)
            return
          

        ### read file ssp
        fid = open(file_SSP,'r')
        
        NProf = int( np.fromfile( fid, float, 1, sep = " " ) )  # number of profiles 
        rProf = np.fromfile( fid, float, NProf, sep = " " )     # range of each profile
        values = np.fromfile (fid, float, -1, sep=" ")          # all the values in .ssp
        n_line = int(len(values)/NProf)                         # number of lines per profile
        cmat = np.zeros((n_line,NProf))                         # sound speed matrix

        for i in range (n_line) :
            cmat[i,:] = values[i*NProf:(i+1)*NProf]

        fid.close()


        ### read file env (to have depths)
        file_env = file_SSP[:-4]+'.env'
        fid = open (file_env,'r')
        f = fid.readlines()

        depth = np.zeros(n_line)     # depths corresponding to profile values
        for i in range(n_line):
            data =f[i+5].split()
            depth[i] =data[0]

        fid.close()


        ### plot ssp2D 
        plt.figure(figsize=(17,8))
        for i in range(NProf):
            plt.subplot (1,NProf,i+1)
            plt.plot(cmat[:,i],depth)
            plt.gca().invert_yaxis()
            plt.title('%d km' %rProf[i])

        #plt.savefig('profiles_'+file_SSP[:-4], dpi=100)

        plt.figure(figsize=(12,8))
        plt.pcolormesh(rProf, depth, cmat, shading='gouraud', cmap ='jet')
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label("sound speed (m/s)")
        plt.title ("Range-dependent SSP - "+file_SSP[:-4])
        plt.xlabel("range (km)")
        plt.ylabel("depth (m)")
        plt.contour(rProf, depth,cmat,10,colors='w',linestyles='dotted')

        #plt.savefig('range-dependent SSP_'+file_SSP[:-4], dpi=100)

        


    
    
    
    
    
    
    
    
    
    
    
