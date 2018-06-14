
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action='ignore')




# class for sources
class source(object):
    ''' A source object '''
    def __init__(self, x_s, y_s, e_dx=10., e_c=10., c_b=1500., label='', t_e=0.):
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
        plt.plot(self.x_s/1.e3, self.y_s/1.e3, color='salmon', marker='o', 
                 markersize=20, label=self.label)
        
    def draw_dxdy(self, e_dx, Np=1):
        ''' compute Np realizations of transductor position
        '''
        self.e_dx = e_dx
        self.dx = np.random.randn(Np)*e_dx
        self.dy = np.random.randn(Np)*e_dx
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
    def __init__(self, x, y, e_x=10.e3, e_dt=1., label='receiver'):
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
    def __init__(self, c_b=None, e_c=None, e_min=0.):
        if c_b is not None:
            self.c_b = c_b
            self._map = lambda x: x/self.c_b
        #
        self.e_min = e_min
        #
        self.e_c = e_c
        if e_c is not None:
            # t = x/(c+e)
            self._emap = lambda x: np.maximum(self.e_min, x*self.e_c/self.c_b**2)
            #self._emap = e_min
            
        
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

#def plot_J(func, x, ):


def geolocalize_xtmap(r, sources, pmap, x0=None, clock_drift=True,
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
            x0 = np.zeros((3))
        else:
            x0 = np.zeros((2))
        #x0[0] = 1.e3
    else:
        if not clock_drift:
            x0 = x0[:2]
    x_r0 = x0[0]
    y_r0 = x0[0]
    
    Ns = len(sources)

    # weights
    if clock_drift:
        W = [1./np.array(r.e_x**2),
             1./np.array(r.e_dt**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for s in sources])]
    else:
        W = [1./np.array(r.e_x**2),
             1./np.array([pmap.e_tp(dist_xyb(x_r0, y_r0, s))**2 for s in sources])]
        
    #print(W)
        
    # scaling factor
    #xy_sc = np.maximum(np.abs(x0[0]), np.abs(x0[1]))
    #xy_sc = np.maximum(r.e_x,xy_sc)
    xy_sc = 1.e3 # should find a better rationale and parametrized expression for this
    if clock_drift:
        dt_sc = np.maximum( r.e_dt, np.abs(x0[2]))
        x_sc = np.array([xy_sc, xy_sc, dt_sc])
    else:
        x_sc = np.array([xy_sc, xy_sc])
    #print(x_sc)
    #print(xy_sc)
    #print(dt_sc)

    # normalize x0
    x0 = x0/x_sc
    #print(x0)
    
    def func(x):
        #
        dx0 = (x[0]-x0[0])*xy_sc
        dy0 = (x[1]-x0[1])*xy_sc
        #
        if clock_drift:
            dt = x[2]*dt_sc
            dt0 = x0[2]*dt_sc  # background value            
        else:
            dt = r.dt
            dt0 = r.dt
        #
        dx_s = x[0]*xy_sc-x_s
        dy_s = x[1]*xy_sc-y_s
        #
        _d = np.sqrt( dx_s**2 + dy_s**2 )
        _t = (r.t_r_tilda - dt - t_e) # propagation time
        #
        J = ( dx0**2 + dy0**2 ) *W[0]
        if clock_drift:
            J += (dt-dt0**2)**2 *W[1]
            J += np.mean( (_t - pmap.t(_d))**2 *W[2] )
        else:
            J += np.mean( (_t - pmap.t(_d))**2 *W[1] )
        return J
        
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
    y = res.x[1]*xy_sc
    if clock_drift:
        dt = res.x[2]*dt_sc
    else:
        dt = r.dt   
    success = res.success
    message = res.message
    
    return x, y, dt, success, message, res 


#----------------------------------------------------------------------------------------------------

def geolocalize(r, sources, x0=None, disp=True):

    # source float position (known)
    x_s = np.array([s.x_s for s in sources])
    y_s = np.array([s.y_s for s in sources])
    # transductor positions (unknown)
    x_t = np.array([s.x_t[0] for s in sources])
    y_t = np.array([s.y_t[0] for s in sources])
    # background velocity
    c_b = np.array([s.c_b for s in sources])    
    # emission time
    t_e = np.zeros_like(x_s)
    # measured arrival times
    t_r_tilda = t_e + np.sqrt((r.x-x_t)**2+(r.y-y_t)**2)/np.array([s.c[0] for s in sources]) \
                - r.dt

    Ns = len(sources)
    idx = slice(3, 3+2*Ns, 2)
    idy = slice(4, 3+2*Ns, 2)

    # weights
    W = [1./np.array(r.e_x**2),
         1./np.array(r.e_dt**2),
         1./np.array([s.e_dx**2 for s in sources]), 
         1./np.array([s.e_c**2 for s in sources])]

    # default background guess
    if x0 is None:
        x0 = np.zeros((3+3*Ns))
        x0[0] = 1.e3
    
    def func(x, W):
        dt = x[2]
        dx = x[idx]
        dy = x[idy]
        # background values
        dt0 = x0[2]
        dx0 = x0[idx]
        dy0 = x0[idy]
        #
        _d = np.sqrt((x[0]-x_s-dx)**2 + (x[1]-y_s-dy)**2)
        _t = (t_r_tilda + dt - t_e)
        #
        J = ( (x[0]-x0[0])**2 + (x[1]-x0[1])**2 )*W[0]
        J += (dt-dt0)**2*W[1]
        J += np.mean( ( (dx-dx0)**2+ (dy-dy0)**2 )*W[2] )
        J += np.mean( ( _d/_t - c_b )**2 *W[3] )
        return J

    def jac(x, W):
        dt = x[2]
        dx = x[idx]
        dy = x[idy]
        # background values
        dt0 = x0[2]
        dx0 = x0[idx]
        dy0 = x0[idy]
        #
        _d = np.sqrt((x[0]-x_s-dx)**2 + (x[1]-y_s-dy)**2)
        _t = (t_r_tilda + dt - t_e)
        #
        jac = np.zeros_like(x0)
        jac[0] = 2.*(x[0]-x0[0])*W[0] + np.mean( 2.*(x[0]-x_s-dx)/_d/_t*( _d/_t - c_b ) *W[3] )
        jac[1] = 2.*(x[1]-x0[1])*W[0] + np.mean( 2.*(x[1]-y_s-dy)/_d/_t*( _d/_t - c_b ) *W[3] )
        jac[2] = 2.*(dt-dt0)*W[1] + np.mean( -2.*_d/_t**2*( _d/_t - c_b ) *W[3] )
        jac[idx] = 2.*(dx-dx0)*W[2] + np.mean( -2.*(x[0]-x_s-dx)/_d/_t*( _d/_t - c_b ) *W[3] )
        jac[idy] = 2.*(dy-dy0)*W[2] + np.mean( -2.*(x[1]-y_s-dy)/_d/_t*( _d/_t - c_b ) *W[3] )
        return jac
    
    # no jacobian, 'nelder-mead' or 'powell'
    #res = minimize(func, x0, args=(W,), method='nelder-mead', options={'maxiter': 10000, 'disp': disp})    
    # with jacobian
    res = minimize(func, x0, args=(W,), jac=jac, method='BFGS', options={'maxiter': 1000, 'disp': disp})    
        
    # extract the solution
    x = res.x[0]
    y = res.x[1]
    dt = res.x[2]
    dx = res.x[idx]
    dy = res.x[idy]
    
    success = res.success
    message = res.message        
    
    return x, y, dt, dx, dy, success, message, res



#--------------------------------------------------------------------------------------------


def geolocalize_xydt(r, sources, x0=None, disp=True):

    # source float position (known)
    x_s = np.array([s.x_s for s in sources])
    y_s = np.array([s.y_s for s in sources])
    # transductor positions (unknown)
    x_t = np.array([s.x_t[0] for s in sources])
    y_t = np.array([s.y_t[0] for s in sources])
    # background velocity
    c_b = np.array([s.c_b for s in sources])    
    # emission time
    t_e = np.zeros_like(x_s)
    # measured arrival times
    t_r_tilda = t_e + np.sqrt((r.x-x_t)**2+(r.y-y_t)**2)/np.array([s.c[0] for s in sources]) \
                - r.dt

        
    Ns = len(sources)

    # weights
    W = [1./np.array(r.e_x**2),
         1./np.array(r.e_dt**2),
         1./np.array([s.e_c**2 for s in sources])]
         #1./np.array([s.e_dx**2 for s in sources])] 

    # default background guess
    if x0 is None:
        x0 = np.zeros ((3))
        x0[0] = 1.e3
    
    def func(x, W):
        dt = x[2]
        # background values
        dt0 = x0[2]
        #
        _d = np.sqrt((x[0]-x_s)**2 + (x[1]-y_s)**2)
        _t = (t_r_tilda + dt - t_e)
        #
        J = ( (x[0]-x0[0])**2 + (x[1]-x0[1])**2 )*W[0]
        J += (dt-dt0)**2*W[1]
        J += np.mean( ( _d/_t - c_b )**2 *W[2] )
        return J

    def jac(x, W):
        dt = x[2]
        # background values
        dt0 = x0[2]
        #
        _d = np.sqrt((x[0]-x_s)**2 + (x[1]-y_s)**2)
        _t = (t_r_tilda + dt - t_e)
        #
        jac = np.zeros_like(x0)
        jac[0] = 2.*(x[0]-x0[0])*W[0] + np.mean( 2.*(x[0]-x_s)/_d/_t*( _d/_t - c_b ) *W[2] )
        jac[1] = 2.*(x[1]-x0[1])*W[0] + np.mean( 2.*(x[1]-y_s)/_d/_t*( _d/_t - c_b ) *W[2] )
        jac[2] = 2.*(dt-dt0)*W[1] + np.mean( -2.*_d/_t**2*( _d/_t - c_b ) *W[2] )
        return jac
    
    # no jacobian, 'nelder-mead' or 'powell'
    #res = minimize(func, x0, args=(W,), method='nelder-mead', options={'maxiter': 10000, 'disp': disp})    
    # with jacobian
    res = minimize(func, x0, args=(W,), jac=jac, method='BFGS', options={'maxiter': 1000, \
                                                                         'disp': disp, 'return_all': True})    
        
    # extract the solution
    x = res.x[0]
    y = res.x[1]
    dt = res.x[2]
    success = res.success
    message = res.message        
    
    return x, y, dt, success, message, res 



#----------------------------------------------------------------------------------------------------

def geolocalize_hard(r, sources, x0=None, disp=True):

    # source float position (known)
    x_s = np.array([s.x_s for s in sources])
    y_s = np.array([s.y_s for s in sources])
    # transductor positions (unknown)
    x_t = np.array([s.x_t[0] for s in sources])
    y_t = np.array([s.y_t[0] for s in sources])
    # emission time
    t_e = np.zeros_like(x_s)
    # measured arrival times
    t_r_tilda = t_e + np.sqrt((r.x-x_t)**2+(r.y-y_t)**2)/np.array([s.c[0] for s in sources]) \
                - r.dt

    Ns = len(sources)
    idx = slice(3, 3+2*Ns, 2)
    idy = slice(4, 3+2*Ns, 2)
    idc = slice(3+2*Ns,3+3*Ns)

    # weights
    W = [1./np.array(r.e_x**2),
         1./np.array(r.e_dt**2),
         1./np.array([s.e_dx**2 for s in sources]), 
         1./np.array([s.e_c**2 for s in sources])]        

    # default background guess
    if x0 is None:
        x0 = np.zeros((3+3*Ns))
        x0[0] = 1.e3
    
    def func(x, W):
        dt = x[2]
        dx = x[idx]
        dy = x[idy]
        dc = x[idc]
        # background values
        dt0 = x0[2]
        dx0 = x0[idx]
        dy0 = x0[idy]
        dc0 = x0[idc]        
        return ( (x[0]-x0[0])**2 + (x[1]-x0[1])**2 )*W[0] \
                + (dt-dt0)**2*W[1] \
                + np.mean( ( (dx-dx0)**2+ (dy-dy0)**2 )*W[2] ) \
                + np.mean( (dc-dc0)**2*W[3] )

    def jac(x, W):
        dt = x[2]
        dx = x[idx]
        dy = x[idy]
        dc = x[idc]
        # background values
        dt0 = x0[2]
        dx0 = x0[idx]
        dy0 = x0[idy]
        dc0 = x0[idc]        
        #
        jac = np.zeros_like(x)
        jac[0] = 2.*(x[0]-x0[0])*W[0]
        jac[1] = 2.*(x[1]-x0[1])*W[0]
        jac[2] = 2.*(dt-dt0)*W[1]
        jac[idx] = 2.*(dx-dx0)*W[2]
        jac[idy] = 2.*(dy-dy0)*W[2]
        jac[idc] = 2.*(dc-dc0)*W[3]
        return jac
    
    # add constraints
    cons = []
    for i, s in enumerate(sources):
        # ! late binding gotcha !
        def cfun(x, i=i, s=s):
            dt = x[2]
            dx = x[idx]
            dy = x[idy]
            dc = x[idc]
            return np.array([(x[0] - s.x_s - dx[i])**2 + (x[1] - s.y_s - dy[i])**2 
                              - (s.c_b + dc[i])**2 *(t_r_tilda[i] + dt - t_e[i])**2])
        # ! late binding gotcha !
        def cjac(x, i=i, s=s):
            dt = x[2]
            dx = x[idx]
            dy = x[idy]
            dc = x[idc]
            #
            jac = np.zeros_like(x)
            jac[0] = 2.*(x[0] - s.x_s - dx[i])
            jac[1] = 2.*(x[1] - s.y_s - dy[i])
            jac[2] = -2.*(s.c_b + dc[i])**2 * (t_r_tilda[i] + dt - t_e[i])
            jac[idx][i] = -2.*(x[0] - s.x_s - dx[i])
            jac[idy][i] = -2.*(x[1] - s.y_s - dy[i])
            jac[idc][i] = - 2.*(s.c_b + dc[i]) * (t_r_tilda[i] + dt - t_e[i])**2
            return jac
        #
        cons.append({'type': 'eq', 'fun' : cfun, 'jac' : cjac})
        
    #'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'func': None, 'maxiter': 100, 'ftol': 1e-06}
    ftol = 1e-06
    print('ftol = %.e' %ftol)
    res = minimize(func, x0, args=(W,), jac=jac, constraints=cons, method='SLSQP', 
                   options={'maxiter': 1000, 'disp': disp, 'eps': 1.4e-08, 'ftol': ftol})    
        
    # extract the solution
    x = res.x[0]
    y = res.x[1]
    dt = res.x[2]
    dx = res.x[idx]
    dy = res.x[idy]
    dc = res.x[idc]
    
    success = res.success
    message = res.message
    
    
    # hard constraints verified ? 
    
    for i, s in enumerate(sources):
        aa = (x - s.x_s - dx[i])**2 + (y - s.y_s - dy[i])**2 - (s.c_b + dc[i])**2 *(t_r_tilda[i] + dt - t_e[i])**2
        print( 'source %d : %.1f' %(i+1,aa))
        #
        #print(' source %d : %.1f' %(i+1, cons[i]['fun'](res.x)))
        print(cons[i]['jac'](res.x))
        
    
    return x, y, dt, dx, dy, dc, success, message, res


