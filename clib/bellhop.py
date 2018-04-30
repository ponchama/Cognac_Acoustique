from pathlib import Path
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.io as sio
from scipy import interpolate
import mpl_toolkits.mplot3d.art3d as art3d

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
                       'zs': 100., 'zmin': 0., 'zmax': zmax, \
                       'rmin':0., 'rmax': rmax, 'NDepth': zmax + 1., \
                       'NRange': rmax * 100. + 1., 'zbox': zmax + 500., 'rbox': rmax + 1.,\
                       'ALimites': [-15., 15.], 'NBeams' : 0, 'bottom':1600., 'file_type': 'R'}
       
        self.params.update(kwargs)
        self.params.update(NDepth = self.params['zmax'] + 1.)
        self.params.update(NRange = self.params['rmax']*100 + 1.)
        self.params.update(zbox = self.params['zmax'] + 500.)
        self.params.update(rbox = self.params['rmax'] + 1.)
        
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
                c, depth, s, lon, lat, h = self.compute_SSP_from_flow(**item)
                self.SSP[ssp_key] = {'c': c[::-1,...].T, 'depth': depth[::-1,...], 
                                     's': s, 'lon': lon, 'lat': lat, 'h': h}
       

   
            
    def compute_SSP_from_flow(self, file=None, datadir=None, hgrid_file=None, \
                              lon=None, lat=None, i_eta=None, i_xi=None, L=None, \
                              itime=0, plot_map=False, contour=True, **kwargs):
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
        
        section=False
        if lon is None or lat is None:
            T = ds['temp'].isel(eta_rho = i_eta, xi_rho = i_xi)
            S = ds['salt'].isel(eta_rho = i_eta, xi_rho = i_xi)            
        else:
            # xarray way, too slow at the moment
            #d = (lon-ds['lon_rho'])**2 + (lat-ds['lat_rho'])**2
            #T = ds['temp'].where(d == d.min(), drop=True)
            #S = ds['salt'].where(d == d.min(), drop=True)
            # numpy way, fast
            if isinstance(lon, float) and isinstance(lat, float):
                d = (lon-grd['lon_rho'])**2 + (lat-grd['lat_rho'])**2
                i_eta, i_xi = np.unravel_index(d.argmin(), d.shape)
                #
                T = ds['temp'].isel(eta_rho = i_eta, xi_rho = i_xi)
                S = ds['salt'].isel(eta_rho = i_eta, xi_rho = i_xi)
            else:
                section=True
                d = (lon[0]-grd['lon_rho'])**2 + (lat[0]-grd['lat_rho'])**2
                i0_eta, i0_xi = np.unravel_index(d.argmin(), d.shape)
                #
                d = (lon[1]-grd['lon_rho'])**2 + (lat[1]-grd['lat_rho'])**2
                i1_eta, i1_xi = np.unravel_index(d.argmin(), d.shape)
                #
                if i1_eta>i0_eta:
                    i_eta = slice(i0_eta, i1_eta+1)
                else:
                    i_eta = slice(i0_eta, i1_eta-1,-1)
                if i1_xi>i0_xi:
                    i_xi = slice(i0_xi, i1_xi+1)
                else:
                    i_xi = slice(i0_xi, i1_xi-1,-1)
                #i_eta = slice(i0_xi, i1_xi, np.sign(i1_xi-i0_xi))
                #i_xi = slice(i0_xi, i1_xi, np.sign(i1_xi-i0_xi))
                #
                #T = ds['temp'].isel(eta_rho = i_eta, xi_rho = i_xi)
                #S = ds['salt'].isel(eta_rho = i_eta, xi_rho = i_xi)
                
        # compute depth level positions
        h = grd['h'].isel(eta_rho = i_eta, xi_rho = i_xi)
        zeta = ds['zeta'].isel(eta_rho = i_eta, xi_rho = i_xi)
        z = grd.get_z(zeta, h)
        # build a uniform grid, output depth levels, should be an option
        if z.ndim == 3:
            z_uni = z.isel(eta_rho = 0, xi_rho = 0).values
            #z_uni[-1,:,:] = 0.
        elif z.ndim == 1:
            z_uni = z.values
            #z_uni[-1] = 0.
                
        if not section:
            T_uni = interp2z(z_uni, z.values, T.values, extrap=True).squeeze()
            S_uni = interp2z(z_uni, z.values, S.values, extrap=True).squeeze()
            #T_uni = interp2z_1d(z_uni[:,[0]], z[:,0], T.values)
            #S_uni = interp2z_1d(z_uni[:,[0]], z[:,0], S.values)
            #lat = grd.lat_rho[i_eta,i_xi]
            c = get_soundc(T_uni, S_uni, z_uni, lon, lat).squeeze()
            #
            slon, slat, s = lon, lat, None
        else:
            # number of horizontal points in the section
            Np = int(np.sqrt( float(i1_eta-i0_eta)**2 + float(i1_xi-i0_xi)**2 ))
            S = np.arange(Np+1).astype(float)/Np
            c_s = np.zeros((z_uni.shape[0],Np))
            h_s = np.zeros((Np))
            print('The transect will contain %d horizontal points' %Np)
            #
            xx, yy = np.meshgrid(np.arange(2.),np.arange(2.))
            for i, s in enumerate(S[:-1]):
                s_eta = i0_eta + s *np.abs(i1_eta-i0_eta)
                i_eta = int(s_eta)
                di_eta = int(s *np.abs(i1_eta-i0_eta))
                #
                s_xi = i0_xi + s *np.abs(i1_xi-i0_xi)
                i_xi = int(s_xi)
                di_xi = int(s *np.abs(i1_xi-i0_xi))
                #
                lz = z.isel(eta_rho = slice(di_eta, di_eta+2), xi_rho = slice(di_xi, di_xi+2))
                lT = ds['temp'].isel(eta_rho = slice(i_eta,i_eta+2), xi_rho = slice(i_xi,i_xi+2))
                lS = ds['salt'].isel(eta_rho = slice(i_eta,i_eta+2), xi_rho = slice(i_xi,i_xi+2))
                #
                T_uni = interp2z(z_uni, lz.values, lT.values, extrap=True).squeeze()
                S_uni = interp2z(z_uni, lz.values, lS.values, extrap=True).squeeze()
                lc = get_soundc(T_uni, S_uni, z_uni[:,None,None], lon, lat).squeeze()
                h_s[i] = interpolate.interp2d(xx, yy, -lz.isel(s_rho=0).values, kind='linear')(s_xi-i_xi, s_eta-i_eta)
                for k in range(z_uni.size):
                    c_s[k,i] = interpolate.interp2d(xx, yy, lc[k,...], kind='linear')(s_xi-i_xi, s_eta-i_eta)
                print('%d/%d' %(i,Np))
            #
            lon_s = lon[0] + s*lon[1]
            lat_s = lat[0] + s*lat[1]
            s = S * earth_dist(lon,lat)
        
        #
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
            im = ax.pcolormesh(grd['lon_rho'],grd['lat_rho'],toplt,
                               vmin=toplt.min(),vmax=toplt.max(), 
                               cmap=cmap)
            cbar = plt.colorbar(im, format='%.1f', extend='both')
            plt.plot(lon, lat, '*', markeredgecolor='white', markerfacecolor='cadetblue', markersize=20)
            if section:
                plt.plot(lon, lat, '-', color='cadetblue', linewidth=2)
            ax.set_title('surface temperature [degC]')
           
        if contour :
            
            cp = plt.contour(grd['lon_rho'], grd['lat_rho'], grd['h'],[500,1000,2500],colors='white')
            plt.clabel(cp, inline=1, fmt='%d', fontsize=10)
                
        return c_s, -z_uni, s, lon_s, lat_s, h_s 
            
    
          
        
        
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
        if np.ndim(self.SSP[ssp_key]['c']) == 1 : 
            c = self.SSP[ssp_key]['c'][:]
        else : 
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
            f.write('%.1f %.1f %.1f %.1f %.1f %.1f\n'%(depth_max,self.params['bottom'],10,2,0.1,0)) #gravier
            #
            f.write('%d \t  !NSD\n' %1)
            f.write('%d / \t  !Source depth\n' % self.params['zs'])
            
            f.write('%d \t  !Number receiver depth\n' % self.params['NDepth'])
            f.write('%.1f %.1f / \t  !Receiver depths\n' %(self.params['zmin'], self.params['zmax']))     
            f.write('%d \t  !Number of ranges\n' % self.params['NRange'])            
            f.write('%.1f  %.1f / \t  !Range limits\n' %(self.params['rmin'], self.params['rmax']))
                       
            f.write('\'%s\'\n' %self.params['file_type'])  # R : .ray ; IB : .shd ; A : .arr
            f.write('%d  \t  !NBeams\n'%self.params['NBeams'])      #! Bellhop calcule le nombre optimal de rayons
            f.write('%.1f  %.1f  / \t  !Angles limites\n' %( 
                self.params['ALimites'][0],self.params['ALimites'][1]))
            f.write('%.1f  %.1f  %.1f \t  !Steps zbox(m) rbox(km)\n' %(
                0.0,self.params['zbox'],self.params['rbox']))

        
    
    
    def generate_sspfile(self, ssp_key, file_env=None, SSP_depth_step=1):
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
            
        # name of ssp file
        file_ssp = file_env[:-4]+'.ssp'
        print('Output file is : '+file_ssp)
            
        # load sound speed profile
        r = self.SSP[ssp_key]['s']  # range
        c = self.SSP[ssp_key]['c']
        depth = self.SSP[ssp_key]['depth'][:]

        # Create environment file
        with open(file_ssp, 'w') as f:
            f.write('%d\n' %np.shape(r)[0])
            for i in range (len(r)): 
                f.write('%.1f ' % r[i])
            f.write('\n')
                
            for i in range(0,len(depth),SSP_depth_step):
                for j in range (len(r)):
                    f.write('%.1f ' % c[j,i])
                f.write('\n')

    
    
    def read_arrivals_asc (self, filename = None, Narrmx = 50): 
        ''' read the arrival time/amplitude data computed by Bellhop
        
        Parameters
        ----------
        filename: str
            Name of the Arrivals file (.arr)
        Narrmx : int
            Maximum number of arrivals allowed
            
        Out
        ----------
        Arr : dic
            Contains all the arrivals information
        Pos : dic
            Contains the positions of source and receivers
            
        '''
        
        if filename is None :
            filename = self.params['name']+'.arr'
            
        file = Path("./%s" %filename)
        if not file.is_file():
            print("Le fichier %s n'exite pas dans ce répertoire." %filename)
            return
        
        Pos = {}
        Arr = {}
        # open the file
        fid = open(filename,'r')
        
        # read the header info
        theline = str( fid.readline() )
        data = theline.split()
        freq  = float( data[0] )
        Nsd = int( data[1] )     # number of source depths
        Nrd = int( data[2] )     # number of receiver depths
        Nrr = int( data[3] )     # number of receiver ranges
        
        theline = str( fid.readline() )
        datai = theline.split()
        Pos['s'] = {'depth': [float(datai[i]) for i in range(len(datai))]}
        theline = str( fid.readline() )
        data_d = theline.split()
        theline = str( fid.readline() )
        data_r = theline.split() 
        Pos['r'] = {'depth': [float(data_d[i]) for i in range(len(data_d))], \
                    'range': [float(data_r[i]) for i in range(len(data_r))]}
         
        
        # loop to read all the arrival info (delay and amplitude)

        A         = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # complex pressure amplitude of each ray at the receiver
        delay     = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # travel time in seconds from source to receiver
        SrcAngle  = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # angle at which the ray left the source (pos is down, neg is up)
        RcvrAngle = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # angle at which the ray is passing through the receiver
        NumTopBnc = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # number of surface bounces
        NumBotBnc = np.zeros( (Nrr, Narrmx, Nrd, Nsd) )   # number of bottom bounces
        Narr      = np.zeros( (Nrr, Nrd, Nsd) )           # number of arrivals
        
        
        for isd in range (Nsd):
            Narrmx2 = int ( fid.readline() ) # max. number of arrivals to follow
            print ('Max.number of arrivals for source index %d is %d' %(isd, Narrmx2))
            for ird in range (Nrd) : 
                #print('receiver depth number %d/%d' %(ird+1, Nrd))
                for ir in range (Nrr):
                    narr = int ( fid.readline() )   # number of arrivals
                    if narr > 0 :                   # do we have any arrivals ? 
                        
                        da = np.zeros((narr,8))
                        for i in range (narr): 
                            theline = str( fid.readline() )
                            datai = theline.split()
                            da[i,:] = datai
                        
                        narr = min(narr,Narrmx)  #we'll keep no more than Narrmx values
                        Narr [ir, ird, isd] = narr
                        
                        A [ir, :narr, ird, isd] = da [:narr,0] * np.exp (1j * da[:narr,1]*np.pi/180)
                        delay [ir, :narr, ird, isd] = da[:narr,2] + 1j * da[:narr,3]
                        SrcAngle [ir, :narr, ird, isd] = da[:narr,4]
                        RcvrAngle [ir, :narr, ird, isd] = da[:narr,5]
                        NumTopBnc [ir, :narr, ird, isd] = da[:narr,6]
                        NumBotBnc [ir, :narr, ird, isd] = da[:narr,7]
        
        Arr = {'A':A, 'delay':delay, 'SrcAngle':SrcAngle, 'RcvrAngle':RcvrAngle, \
               'NumTopBnc': NumTopBnc, 'NumBotBnc':NumBotBnc, 'Narr' : Narr}    
        
        fid.close()  
        return Arr, Pos 
    
    
    
    def plotarr(self, Arr, Pos, irr=0, ird=0, isd=0, filename = None,):
        ''' plot the arrivals calculated by Bellhop
        
        Parameters
        ----------
        filename: str
            Name of the Arrivals file (.arr)
        irr : int
            Index of receiver range
        ird : int
            Index of receiver depth
        isd : int
            Index of source depth 
        '''
        
        # read
        #Narrmx = 5000
        #Arr, Pos = self.read_arrivals_asc(filename, Narrmx)
        
        # stem plot for a single receiver
        plt.figure()
        Narr = int(Arr['Narr'][irr, ird, isd])

        for i in range (Narr) : 
            markerline, stemlines, baseline = plt.stem( [Arr['delay'][irr, i, ird, isd]],\
                                               [abs(Arr['A'][irr, i, ird, isd])])

            if np.logical_and ( Arr['NumTopBnc'][irr, i,ird,isd] == 0, Arr['NumBotBnc'][irr, i,ird,isd] == 0):
                plt.setp(stemlines, color = 'r')
                plt.setp(markerline, color = 'r')
        
            elif Arr['NumBotBnc'][irr, i,ird,isd] == 0:
                plt.setp(stemlines, color = 'b')
                plt.setp(markerline, color = 'b')
        
            else : 
                plt.setp(stemlines, color = 'k')
                plt.setp(markerline, color = 'k')
 
        plt.ylim(ymin=0)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Source depth : %.1fm -- Receiver depth : %.1fm -- Receiver range : %.1fkm' \
                  % (Pos['s']['depth'][isd], Pos['r']['depth'][ird], Pos['r']['range'][irr]/1000.))
        
 
    
    
         # depth-time stem plot

#        fig = plt.figure()
#        ax = fig.add_subplot(1, 1, 1, projection='3d')
#        for ird1 in range (np.shape(Arr['A'])[2]):
#            Narr = int(Arr['Narr'][irr,ird,isd])
#            x = Arr['delay'][irr,:Narr,ird,isd]
#            y = Pos['r']['depth'][ird1]*np.ones_like(x)
#            z = abs(Arr['A'][irr, :Narr, ird, isd])
#
#            for xi, yi, zi in zip(x, y, z):        
#                line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1, 1))
#                ax.add_line(line)
#        
#        ax.set_xlim3d(6, 7)
#        ax.set_ylim3d(400, 600)
#        ax.set_zlim3d(0, 1e-4)    
#        
#        plt.xlabel('Time (s)')
#        plt.ylabel('Depth (m)')
#        plt.title ('Source depth : %.1fm -- Receiver range : %.1fkm' \
#                  % (Pos['s']['depth'][isd], Pos['r']['range'][irr]/1000.))
#        plt.show()
            
        
#        # range-time stem plot
                
#        fig1 = plt.figure()
#        ax = fig1.add_subplot(1, 1, 1, projection='3d')
#        for irr in range (np.shape(Arr['A'])[0]):
#            Narr = int(Arr['Narr'][irr,ird,isd])
#            x = Arr['delay'][irr,:Narr,ird,isd]
#            y = Pos['r']['range'][irr]*np.ones_like(x)
#            z = abs(Arr['A'][irr, :Narr, ird, isd])
#
#            for xi, yi, zi in zip(x, y, z):        
#                line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1, 1))
#                ax.add_line(line)
#        
#        ax.set_xlim3d(6, 7)
#        ax.set_ylim3d(9500,10500)
#        ax.set_zlim3d(0, 1e-4)    
#        
#        plt.xlabel('Time (s)')
#        plt.ylabel('Range (m)')
#        plt.title ('Source depth : %.1fm -- Receiver depth : %.1fm' \
#                  % (Pos['s']['depth'][isd], Pos['r']['depth'][ird]))
#        plt.show()
        
        
        
        
        
            
    def plotray (self, filename = None, dist = False):
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

                   ## traveled distance 
                   d = 0
                   for i in range (len(r)-1):
                       d += np.sqrt((r[i+1]-r[i])**2 + (z[i+1]-z[i])**2) 
                        
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
                   if dist :
                        label = '%.2fm' %d
                   else : 
                        label = None
                   plt.plot( r/1000., -z,  color = color, label = label)
                   plt.axis([rmin/1000.,rmax/1000.,-zmax,-zmin])
        
        plt.title(filename[:-4])
        plt.xlabel('range (km)')
        plt.ylabel('profondeur (m)')
        plt.grid()
        
        if dist : 
            plt.legend()
        #plt.savefig('plotray_'+filename[:-4], dpi=100)

        fid.close()

        
        
        
        
        
        
        
    def plotE (self, filename = None, dist = False, plot=True):
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

        SrcAngle  = np.zeros( (Nsz, Nalpha) )   # angle at which the ray left the source (pos is down, neg is up)
        NumTopBnc = np.zeros( (Nsz, Nalpha) )   # number of surface bounces
        NumBotBnc = np.zeros( (Nsz, Nalpha) )   # number of bottom bounces
        Dist      = np.zeros( (Nsz, Nalpha) )   # traveled distance

        for isz in range(Nsz):
            for ibeam in range(Nalpha):
                theline = str( fid.readline() )
                l = len( theline )
                if l > 0:
                    alpha0 = float( theline )
                    theline = str( fid.readline() )
                    datai = theline.split()
                    nsteps    = int( datai[0] )
                    numTopBnc = int( datai[1] )
                    numBotBnc = int( datai[2] )
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
                    
                    
                    if plot : 
                        ## Color of the ray
                        #RED : no reflexion on top and bottom
                        if np.logical_and (numTopBnc==0, numBotBnc==0):
                            color = 'r'
                        #BLUE : no reflexion on bottom
                        elif numBotBnc==0 :
                            color = 'b'
                        #BLACK : reflexion on top and bottom
                        else : 
                            color = 'k'

                        ## plot  
                        if dist :
                            label = '%.2fm' %d
                        else : 
                            label = None

                        plt.plot( r/1000., -z,  color = color, label = label)
                        plt.axis([rmin/1000.,rmax/1000.,-zmax,-zmin])
                    
                   

                    SrcAngle[isz,ibeam]  = alpha0
                    NumTopBnc[isz,ibeam] = numTopBnc
                    NumBotBnc[isz,ibeam] = numBotBnc
                    Dist[isz,ibeam]      = d

                else : 
                    SrcAngle[isz,ibeam]  = np.NaN
                    NumTopBnc[isz,ibeam] = np.NaN
                    NumBotBnc[isz,ibeam] = np.NaN
                    Dist[isz,ibeam]      = np.NaN
                    
        if plot : 
            plt.title(filename[:-4])
            plt.xlabel('range (km)')
            plt.ylabel('profondeur (m)')
            plt.grid()

            if dist : 
                plt.legend()
            #plt.savefig('plotray_'+filename[:-4], dpi=100)


        dictE = {'SrcAngle':SrcAngle[~np.isnan(SrcAngle)], 'NumTopBnc': NumTopBnc[~np.isnan(NumTopBnc)], \
                 'NumBotBnc':NumBotBnc[~np.isnan(NumBotBnc)], 'Dist' : Dist[~np.isnan(Dist)]}    
        fid.close()

        return dictE    
        
        
        
        

       
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
        if np.ndim(self.SSP[ssp_key]['c']) == 1 : 
            c = self.SSP[ssp_key]['c'][:]
        else : 
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
        fid = open(filename,'r')
        
        NProf = int( np.fromfile( fid, float, 1, sep = " " ) )  # number of profiles 
        rProf = np.fromfile( fid, float, NProf, sep = " " )     # range of each profile
        values = np.fromfile (fid, float, -1, sep=" ")          # all the values in .ssp
        n_line = int(len(values)/NProf)                         # number of lines per profile
        cmat = np.zeros((n_line,NProf))                         # sound speed matrix

        for i in range (n_line) :
            cmat[i,:] = values[i*NProf:(i+1)*NProf]

        fid.close()


        ### read file env (to have depths)
        file_env = filename[:-4]+'.env'
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
            plt.title('%d m' %rProf[i])

        #plt.savefig('profiles_'+file_SSP[:-4], dpi=100)

        plt.figure(figsize=(12,8))
        plt.pcolormesh(rProf, depth, cmat, shading='gouraud', cmap ='jet')
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label("sound speed (m/s)")
        plt.title ("Range-dependent SSP - "+filename[:-4])
        plt.xlabel("range (m)")
        plt.ylabel("depth (m)")
        plt.contour(rProf, depth,cmat,10,colors='w',linestyles='dotted')

        #plt.savefig('range-dependent SSP_'+file_SSP[:-4], dpi=100)

        


    
    
    
    
    
    
    
    
    
    
    
