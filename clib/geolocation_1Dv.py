
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action='ignore')




# class for sources
class source(object):
    ''' A source object '''
    def __init__(self, x_s, y_s=0., e_dx=10., e_c=10., c_b=1500., label='', 
                 t_e=0., v_x = 0.,v_y = 0., t0 = 0.):
        '''
        Parameters
        ----------
        x_s, y_s  - horizontal position in meters
        e_i - rms between transductor position uncertainty on the distance estimation
        label - a label for the source
        '''
        self.x_s = x_s
        self.y_s = y_s
        #
        self.e_dx = e_dx
        self.draw_dxdy(e_dx)
        #
        self.c_b = c_b
        self.e_c = e_c
        self.draw_celerity(e_c, c_b=c_b)
        #
        self.t_e = t_e
        #
        self.v_x = v_x
        self.v_y = v_y
        self.t0 = t0
        
        #self.tau_i=None
        self.label = ('source ' + label).strip()
    
    def __getitem__(self, item):
        if item is 'x':
            return self.x_s
        elif item is 'y':
            return self.y_s
        else:
            return getattr(self, item)
    
    def plot(self, t=None):
        if t is None : 
            x, y = self.x_s, self.y_s
            lab = "t = 0h"
        else : 
            x, y = self.get_xy(t)
            lab = "t = %.1f h" %(t/3600.)
            
        plt.plot(x/1.e3, y/1.e3, color='darkorange', marker='o', 
                 markersize=20, label=lab)
        
    def draw_dxdy(self, e_dx, Np=1):
        ''' compute Np realizations of transductor position
        '''
        self.e_dx = e_dx
        self.dx = np.random.randn(Np)*e_dx
        #self.dy = np.random.randn(Np)*e_dx
        self.dy = 0.
        self.x_t = self.x_s + self.dx
        self.y_t = self.y_s + self.dy


    def draw_celerity(self, e_c, Np=1, c_b=None):
        ''' compute Np celerities with rms celerities e_c
            ne sert plus, pourrait être retiré
        '''
        if c_b is None:
            c_b = self.c_b
        else:
            self.c_b = c_b
        self.e_c = e_c
        self.c = c_b + np.random.randn(Np)*e_c
      
    def get_xy(self, t) : 
        return self.x_s + self.v_x * (t - self.t0), self.y_s + self.v_y * (t - self.t0)
    
        
# class for receivers:
class receiver(object):
    ''' A receiver object '''
    def __init__(self, x, y=0., e_x=10.e3, e_dt=0.1, e_v=0.1, 
                 label='receiver', v_x = 0., v_y = 0., t0 = 0.):
        '''
        Parameters
        ----------
        x,y  - horizontal position in meters
        e_dt - uncertainty on the clock drift in seconds
        label - a label for the receiver
        '''
        self.x = x
        self.y = y
        self.e_dt = e_dt
        self.e_x = e_x
        self.draw_clock_drift(e_dt)
        self.label = ('receiver ' + label).strip()

        self.v_x = v_x
        self.v_y = v_y
        self.t0 = t0
        self.e_v = e_v
        
    def __getitem__(self, item):
        return getattr(self, item)
        
    def plot(self, t=None):
        if t is None : 
            x, y = self.x, self.y
        else : 
            x, y = self.get_xy(t)
        plt.plot(x/1.e3, y/1.e3, color='green', marker='*', 
                 markersize=20, label=self.label)
        
    def draw_clock_drift(self, e_dt, Np=1):
        self.e_dt = e_dt
        self.dt = np.random.randn(Np)*e_dt
    
    def get_xy(self, t):
        return  self.x + self.v_x * (t - self.t0), self.y + self.v_y * (t - self.t0)

    
    
# class for space-time mapping and error
class xtmap(object):
    ''' Mapping between distance and time with associated errors
    This mapping can be linear, i.e. determined by a constant value of sound velocity (c_b)
    It may be also provided by a Bellhop simulation output (not implemented)
    Errors on this mapping are required.
    
    Parameters
    ----------
    c_b: float
        Sound speed velocity
    e_c: float
        Error on our sound speed velocity
        
    '''

    def __init__(self, c_b=None, e_c=0., e_t=None, e_min=1.e-3):
        if c_b is not None:
            self.c_b = c_b
            self._map = lambda x: abs(x)/self.c_b
        #
        self.e_c = e_c
        self.e_t = e_t
        self.e_min = e_min
             
        if self.e_t is not None : 
            if type(self.e_t) is float : 
                #self._emap = e_t
                self._emap = lambda x: np.maximum(self.e_min, self.e_t)
            else : #function
                self._emap = self.e_t
        else : 
            self._emap = lambda x: self.e_min
        
        
    def t(self, x):
        ''' Returns time of propagation given a range x
        '''
        return self._map(x)
    
    def draw_t(self, x, Np=1):
        ''' Draws a random value of propagation time 
        '''
        if Np == 1 :
            return self._map(x) + np.random.randn(x.size)*self._emap(x)
        else:
            return self._map(x) + np.random.randn(x.size, Np)*self._emap(x)
    
    def e_tp(self, x):
        ''' Returns, for a given range x, the error on propagation time
        '''
        return self._emap(x)
        

# utils
def dist(a,b):
    return np.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2)

def dist_xyb(x,y,b):
    return np.sqrt((x-b['x'])**2+(y-b['y'])**2)




#----------------------------------------------------------------------------------------------------


def geolocalize_xtmap_1Dv(r, sources, t_e, pmap, x0=None, clock_drift=True, plot_min=False, 
                      method='nelder-mead', options=None, disp=False):
    ''' Find the location of a receiver
    
    Parameters:
    -----------
    ...
    
    '''

    # source float position (known)
    x_s = np.array([s.x_s for s in sources])
    y_s = np.array([s.y_s for s in sources])
    
    # a priori float position
    if x0 is None:
        if clock_drift:
            x0 = np.zeros((3))  # x, v, dt
        else:
            x0 = np.zeros((2))  # x, v
    else:
        if not clock_drift:
            x0 = x0[:2]
   
    
    x_r0 = x0[0]  # position à priori du récepteur
    y_r0 = 0.
    

    Ns = len(sources)
    s = sources[0]

    # weights
    if clock_drift:
        W = [1./np.array(r.e_x**2),
             1./np.array(r.e_v**2),
             1./np.array(r.e_dt**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for t in t_e])]
    else:
        W = [1./np.array(r.e_x**2),
             1./np.array(r.e_v**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for t in t_e])]
    
    # scaling factor
    xy_sc = 1.e3 # should find a better rationale and parametrized expression for this
    v_sc = 0.1
    
    if clock_drift:
        dt_sc = np.maximum( r.e_dt, np.abs(x0[1]))
        x_sc = np.array([xy_sc, v_sc, dt_sc])
    else:
        x_sc = np.array([xy_sc, v_sc])

    # normalize x0
    x0 = x0/x_sc
   
    
    def func(x):
        #
        dx0 = (x[0]-x0[0])*xy_sc
        dv0 = (x[1]-x0[1])*v_sc
        #
        if clock_drift:
            dt = x[2]*dt_sc
            dt0 = x0[2]*dt_sc  # background value            
        else:
            dt = r.dt
            dt0 = r.dt
        #
        _t = (r.t_r_tilda - dt - t_e) # propagation time
        _d = x[0]*xy_sc + x[1]*v_sc*(r.t_r_tilda - dt - r.t0) - s.x_s - s.v_x*(t_e - s.t0)
        
        #
        J = ( dx0**2 ) *W[0] 
        J += ( dv0**2 ) *W[1] 
        if clock_drift:
            J += (dt-dt0**2)**2 *W[2]
            J += np.sum( (_t - pmap.t(_d))**2 *W[3] )
        else:
            J += np.sum( (_t - pmap.t(_d))**2 *W[2] )
            pass
        return J
    
     ##################################################################
    ################ Tracé de la fonction J ##########################
    
    if plot_min : 
        #x_grd = np.linspace(-50., 50., 100.)
        #f_grd = np.array([func(np.array([lx])) for lx in x_grd])
        #plt.plot(x_grd,f_grd)
        #plt.title('Fonction de minimisation')
        #plt.xlabel('x (km)')
        #plt.grid()
        
        x_grd = np.linspace(-30.e3, 30.e3, 601.) / xy_sc
        v_grd = np.linspace(-1,6,71.) / v_sc
        f_grd2D = np.zeros((x_grd.size, v_grd.size))
        X, Y = np.meshgrid(x_grd, v_grd)

        for i in range (len(x_grd)) : 
            for j in range (len(v_grd)) :
                f_grd2D[i,j] = func(np.array([x_grd[i], v_grd[j]]))
        
        np.save('f_grd2D', f_grd2D)
       
        plt.figure(figsize=(8,5))
        plt.pcolormesh(X,Y*v_sc,f_grd2D.T)
        plt.title("Fonction de minimisation", fontsize=16)
        plt.xlabel('x (km)', fontsize=14)
        plt.ylabel('v (m/s)', fontsize=14)
        plt.colorbar()
        plt.clim([0, 30000.])

     ###############################################################
    ###############################################################
        
        
    # no jacobian, 'nelder-mead' or 'powell'
    if method is None:
        method = 'nelder-mead'
        method = 'powell'
    if options is None:
        maxiter = 1000
        if method is 'nelder-mead':
            # default: xatol': 0.0001, 'fatol': 0.0001,
            options = {'maxiter': maxiter, 'disp': disp, 'xatol': 1.e-8,'fatol': 1.e-8}
        elif method is 'powell':
            # default: 'xtol': 0.0001, 'ftol': 0.0001
            options = {'maxiter': maxiter, 'disp': disp, 'xtol': 1.e-8,'ftol': 1.e-8}
            
    # solve
    res = minimize(func, x0, method=method, options=options)

    # rerun if fails after 1000 iterations
    if not res.success:
        print('No convergence, try with 1000 more iterations')
        res = minimize(func, res.x, method=method, options=options)
        if not res.success:
            print('Still no convergence')            
            print(res)
        
    # extract the solution
    x = res.x[0]*xy_sc
    v = res.x[1]*v_sc
    if clock_drift:
        dt = res.x[2]*dt_sc
    else:
        dt = r.dt   
    success = res.success
    message = res.message
    JJ = func([res.x[0],res.x[1],dt])
    
    return x, v, dt, success, message, res, JJ 

# ---------------------------------------------------------------------------------------------
###############################################################################################

### Fonctions de calcul des transects 


def simu (r, sources, Nmc, t_e, t_drift, pmap, x0=None, new_method=False) : 
    ''' It returns rms and bias on x position for one receiver position'''
    x = np.zeros(Nmc)
    v = np.zeros(Nmc)
    dt = np.zeros(Nmc)
    su = np.zeros (Nmc)
    J = np.zeros(Nmc)
    
    for i in range(Nmc):
            
        _t = []
        for t in t_e : 
            x_r, y_r = r.get_xy(t)
            x_s, y_s = sources[0].get_xy(t)
            rg = np.sqrt((x_r-x_s)**2 + (y_r - y_s)**2)
            _t.append(t + pmap.draw_t(rg))

        r.t_r_tilda = np.array(_t+r.dt).squeeze()
        
        if not new_method : 
            x[i], v[i], dt[i], success, message, res, J[i] = geolocalize_xtmap_1Dv(r, sources, t_e, pmap, \
                                                                             clock_drift=t_drift, \
                                                                            x0 = x0)    
        else : 

            x1, v1, dt1, success1, message1, res1, J1 = geolocalize_xtmap_1Dv(r, sources, t_e, pmap, clock_drift=t_drift, \
                                                        x0 = np.array([sources[0].x_s - (r.t_r_tilda - t_e)[0]*1500.,0.,0.]))
            x2,v2, dt2, success2, message2, res2, J2 = geolocalize_xtmap_1Dv(r, sources, t_e, pmap, clock_drift=t_drift, \
                                                        x0 = np.array([sources[0].x_s + (r.t_r_tilda - t_e)[0]*1500.,0.,0.]))
            if J1 == J2 : 
                x[i], v[i], dt[i], success, message, res, J[i] = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
            elif J1 < J2 : 
                x[i], v[i], dt[i], success, message, res, J[i] = x1, v1, dt1, success1, message1, res1, J1
            else : 
                x[i], v[i], dt[i], success, message, res, J[i] = x2, v2, dt2, success2, message2, res2, J2

        if success :
            su[i] = 1 
        elif message.find('iterations')!= -1 : 
            su[i] = 0
            
    # rms error on the receiver position
    d_rms = np.sqrt( np.mean( (x[np.where(su==1)] - r.x)**2 ) )
    # biais on the receiver position
    bias_x = x[np.where(su==1)].mean()-r.x
    return (d_rms, bias_x, su)



from ipywidgets import FloatProgress
from IPython.display import display


def transect (sources, X, Y, Nmc, t_e, pmap, v_x=0.1, clock_drift = False, e_dt=0.01, x0=None, new_method=False) :
    RMS_t = np.zeros((len(X)))
    BiasX_t = np.zeros((len(X)))
    Success_t = np.zeros((Nmc, len(X)))
    
    r = receiver(X[0], Y, e_dt=e_dt)
    r_dt = r.dt
    
    f = FloatProgress(value = 0., min=0., max=100., step=1., orientation='horizontal', description = 'Loading :')
    display(f)
    
    for i in range (len(X)) :
        
        f.value = i/len(X)*100.

        # init a receiver
        r = receiver(X[i], Y, e_dt=e_dt, v_x=v_x)
        #r.dt = r_dt # unchanged variable during simulations 
        #
        d_rms, bias_x, su = simu (r, sources, Nmc, t_e=t_e, t_drift = clock_drift, pmap=pmap, x0=x0, new_method=new_method)

        RMS_t[i]       = d_rms
        BiasX_t[i]     = bias_x
        Success_t[:,i] = su

    f.value = 100.
    
    return RMS_t, BiasX_t, Success_t


