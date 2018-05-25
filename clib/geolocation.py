
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action='ignore')




# class for sources
class source(object):
    ''' A source object '''
    def __init__(self, x_s, y_s, e_dx=10., e_c=10., c_b=1500., label=''):
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


def dist(a,b):
    return np.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2)



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


