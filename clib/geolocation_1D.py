
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action='ignore')




# class for sources
class source(object):
    ''' A source object '''
    def __init__(self, x_s, y_s=0., e_dx=10., e_c=10., c_b=1500., label='', t_e=0.):
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
        #self.tau_i=None
        self.label = ('source ' + label).strip()
    
    def __getitem__(self, item):
        if item is 'x':
            return self.x_s
        elif item is 'y':
            return self.y_s
        else:
            return getattr(self, item)
    
    def plot(self):
        plt.plot(self.x_s/1.e3, self.y_s/1.e3, color='darkorange', marker='o', 
                 markersize=20, label=self.label)
        
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
        '''
        if c_b is None:
            c_b = self.c_b
        else:
            self.c_b = c_b
        self.e_c = e_c
        self.c = c_b + np.random.randn(Np)*e_c
        
        
# class for receivers:
class receiver(object):
    ''' A receiver object '''
    def __init__(self, x, y=0., e_x=10.e3, e_dt=0.1, label='receiver'):
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

    def __getitem__(self, item):
        return getattr(self, item)
        
    def plot(self):
        plt.plot(self.x/1.e3,self.y/1.e3, color='green', marker='*', markersize=20, label=self.label)
        
    def draw_clock_drift(self, e_dt, Np=1):
        self.e_dt = e_dt
        self.dt = np.random.randn(Np)*e_dt


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
    def __init__(self, c_b=None, e_c=None, e_t=None, e_min=1.e-3):
        if c_b is not None:
            self.c_b = c_b
            self._map = lambda x: abs(x)/self.c_b
        #
        self.e_min = e_min
        #
        self.e_c = e_c
        #if e_c is not None:
        #    self._emap = lambda x: np.maximum(self.e_min, abs(x)*self.e_c/self.c_b**2)

        self.e_t = e_t
        if e_t is not None : 
            #self._emap = e_t
            self._emap = lambda x: np.maximum(self.e_min, self.e_t)
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


def geolocalize_xtmap_1D(r, sources, pmap, x0=None, clock_drift=True, plot_min=False, 
                      method='nelder-mead', options=None, disp=False):
    ''' Find the location of a receiver
    
    Parameters:
    -----------
    ...
    
    '''

    # source float position (known)
    x_s = np.array([s.x_s for s in sources])
    y_s = np.array([s.y_s for s in sources])
    # emission time
    t_e = np.array([s.t_e for s in sources])

    
    
    # a priori float position
    if x0 is None:
        if clock_drift:
            x0 = np.zeros((2))  # x, dt
        else:
            x0 = np.zeros((1))  # x
        #x0[0] = 1.e3
    else:
        if not clock_drift:
            x0 = x0[:1]
   
    
    x_r0 = x0[0]  # position à priori du récepteur
    y_r0 = 0.

    Ns = len(sources)

    # weights
    if clock_drift:
        W = [1./np.array(r.e_x**2),
             1./np.array(r.e_dt**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for s in sources])]
    else:
        W = [1./np.array(r.e_x**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for s in sources])]
        
        
    # scaling factor
    #xy_sc = np.maximum(np.abs(x0[0]), np.abs(x0[1]))
    #xy_sc = np.maximum(r.e_x,xy_sc)
    xy_sc = 1.e3 # should find a better rationale and parametrized expression for this
    
    if clock_drift:
        dt_sc = np.maximum( r.e_dt, np.abs(x0[1]))
        x_sc = np.array([xy_sc, dt_sc])
    else:
        x_sc = np.array([xy_sc])
        
      
    #print('x_sc : ', x_sc)
    #print('xy_sc : ', xy_sc)
    #print(dt_sc)

    # normalize x0
    x0 = x0/x_sc
   
    
    def func(x):
        #
        dx0 = (x[0]-x0[0])*xy_sc
        #
        if clock_drift:
            dt = x[1]*dt_sc
            dt0 = x0[1]*dt_sc  # background value            
        else:
            dt = r.dt
            dt0 = r.dt
        #
        dx_s = x[0]*xy_sc-x_s
        #
        _d = dx_s   #np.sqrt( dx_s**2 + dy_s**2 )
        
        #print('dt : ', dt)
        #print('t_e : ', t_e)
        #print('r.t_r_tilda : ', r.t_r_tilda)
        _t = (r.t_r_tilda - dt - t_e) # propagation time
        
        #print('d:', _d)
        #print('t :', _t)
        #print((_t - pmap.t(_d))**2 *W[1])
        
        #
        J = ( dx0**2 ) *W[0]
        if clock_drift:
            J += (dt-dt0**2)**2 *W[1]
            J += np.mean( (_t - pmap.t(_d))**2 *W[2] )
        else:
            J += np.mean( (_t - pmap.t(_d))**2 *W[1] )
            #J = ((_t - pmap.t(_d))**2 *W[1])[1]
            pass
        return J
    
    if plot_min : 
        x_grd = np.linspace(-50., 50., 100.)
        f_grd = np.array([func(np.array([lx])) for lx in x_grd])
        plt.plot(x_grd,f_grd)
        plt.title('Fonction de minimisation')
        plt.xlabel('x (km)')
        plt.grid()

        
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
    #res = minimize(func, x0, args=(W,), method=method, options=options)
    res = minimize(func, x0, method=method, options=options)

    # rerun if fails after 1000 iterations
    if not res.success:
        print('No convergence, try with 1000 more iterations')
        res = minimize(func, res.x, method=method, options=options)
        if not res.success:
            print('Still no convergence')            
            print(res)
        #print(res['message'])
        
    # extract the solution
    x = res.x[0]*xy_sc
    if clock_drift:
        dt = res.x[1]*dt_sc
    else:
        dt = r.dt   
    success = res.success
    message = res.message
    
    return x, dt, success, message, res 

# ---------------------------------------------------------------------------------------------

