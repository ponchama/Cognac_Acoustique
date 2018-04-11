

import scipy.io as sio


def generate_bellhop_envfile_SSP(dir_env, file_SSP, file_bathy, Issp=0):

    '''
    
    Parameters
    ----------
    dir_env: str
        Directory where file will be created
    file_SSP: str
        File containing sound speed profile
    file_bathy: str
        Bathymetric file
    Issp : int
        Sound speed profile index in file_SSP
    
    
    '''

    # file_env needs to be consistent with bathymetric file
    file_env = file_bathy[:-4]+'.env'
    print('Output file is : '+file_env)

    # load sound speed profile

    # file_SSP = 'SSP_4profils.mat'
    SSP = sio.loadmat(file_SSP) # SSP_Dr1m et Depthr
    SSP_Dr1m = SSP['SSP_Dr1m'][Issp,:]
    Depthr = SSP['Depthr'][0,:]

    # Create environment files
    #cd(dirEnv)

    freq = 8000         # frequence emission en Hz
    zs = 50             # profondeur de la source en m 
    zmax = 2550         # profondeur max en m
    rmax = 35           # portee max en km
    NDepth = zmax + 1   # grille en profondeur
    NRange = rmax * 100 + 1 # grille en portee
    ALimites = [-90, 90] # angles d'emission en degres

    with open(dir_env+file_env, 'w') as f:
        f.write('\'Range dep, Gaussian beams\'\n')
        f.write('%.1f\n' %freq)
        f.write('%.d\n' %1)
        f.write('\'SVWT\'\n')
        #
        depth = Depthr.max() # ! Bathymetrie maximale
        f.write('%d %.1f %.1f\n'%(0, 0.0, depth))
        for i in range(0,len(Depthr),5):
            f.write('%.1f %.1f /\n' %(Depthr[i], SSP_Dr1m[i]))
        #
        f.write('\'A*\' %.1f\n' %0.0)
        f.write('%.1f %.1f %.1f %.1f %.1f %.1f\n'%(depth,1800,10,2,0.1,0)) #gravier
        #
        f.write('%d \t  !NSD\n' %1)
        f.write('%d / \t  !Source depth\n' %zs)
        f.write('%d \t  !Number receiver depth\n' %NDepth)
        f.write('%.1f %.1f / \t  !Receiver depths\n' %(0.0, zmax))
    
        f.write('%d \t  !Number of ranges\n' %NRange)
        f.write('%.1f  %.1f / \t  !Range limites\n'%(0, rmax))
        f.write('\'IB\'\n')            #! On choisit de travailler en champ incoherent
        f.write('%d  \t  !NBeams\n'%0)      #! Bellhop calcule le nombre optimal de rayons
        f.write('%.1f  %.1f  / \t  !Angles limites\n' %(ALimites[0],ALimites[1]))
        f.write('%.1f  %.1f  %.1f \t  !Steps zbox(m) rbox(km)\n' %(0.0,zmax+450,rmax+1))


if __name__ == '__main__':

    dir_env = './'
    file_SSP = 'SSP_4profils.mat'
    file_bathy = 'cognac_2000m.bty'
    Issp = 0

    generate_bellhop_envfile_SSP(dir_env, file_SSP, file_bathy, Issp=Issp)



