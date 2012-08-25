import pyqtgraph as pg
import numpy as np
import collections
c = 1.0



class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, v0=0.0, t0=0.0, prog=None, pen=None, brush=None):
        if pen is None:
            pen = pg.intColor(Clock.nClocks, 12)
        if brush is None:
            brush = (0,0,150)
        Clock.nClocks += 1
        self.x0 = x0
        self.v0 = v0
        self.t0 = t0
        self.prog = prog
        self.pen = pen
        self.brush = brush

    def startSim(self, nPts):
        self.data = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float)])
        self.i = 0
        self.x = self.x0
        self.v = self.v0
        self.t = 0.0           ## reference clock always starts at 0
        self.pt = self.t0      ## proper time starts at t0
        
    def advance(self, dt):
        gamma = (1.0 - self.v**2 / c**2) ** -0.5
        g = self.acceleration()
        self.v += g * dt / gamma
        if self.v >= 1.0:
            self.v = 0.999999999999   ## cheating. Use better integration method.
        self.x += self.v * dt
        self.t += dt
        self.pt += dt / gamma
        
        self.data[self.i] = (self.x, self.t, self.v, self.pt)
        self.i += 1
    
    def acceleration(self):
        if self.prog is None:
            return 0.0
        
        ret = 0.0
        for t,a in self.prog:
            if self.t >= t:
                ret = a
        return ret
        
    def getCurve(self, ref=None):
        print self, ref
        if ref is None:
            x = self.data['x']
            y = self.data['t']
            t = self.data['pt']
        else:
            
            ## Transform this worldline into the perspective of another clock.
            
            ## Here's what we'll do:
            ## For each time step:
            ##    - transform the entire worldline to the reference frame
            ##    - find the x where the worldline crosses t=0
            ##    - append a new point onto the transformed line
            worldline = np.vstack([self.data['x'], self.data['t'], np.ones(len(self.data['t']))])
            points = ([], [], [])
            
            for i in range(len(self.data['t'])):
                ## translate reference clock to origin
                translate = np.array([[1,0,-ref.data['x'][i]],[0,1,-ref.data['t'][i]]])
                
                ## lorentz-transform into reference frame
                v = ref.data['v'][i]
                gamma = (1.0 - v**2 / c**2) ** -0.5
                vg = -v/c**2 * gamma
                lorentz = np.array([[gamma, vg], [vg, gamma]])
                
                ## transform the worldline
                transform = np.dot(lorentz, translate)
                wl2 = np.dot(transform, worldline)
                x = wl2[0]
                t = wl2[1]
                
                ## find zero-crossing
                ind = zeroCross(t)
                if ind is None:
                    points[0].append(np.nan)
                    points[1].append(np.nan)
                    points[2].append(np.nan)
                else:
                    points[0].append(x[ind])
                    points[1].append(ref.data['pt'][i])
                    points[2].append(self.data['pt'][ind])
                
            x = np.array(points[0])
            y = np.array(points[1])
            t = np.array(points[2])
            
            #x = gamma * (self.data['x'] - v*self.data['t'] - ref.data['x'] + v*ref.data['t'])
            #y = gamma * (self.data['t'] - v*self.data['x']/c**2 + v*ref.data['x']/c**2)
            
            ##x = self.data['x'] - ref.data['x']
            ##y = self.data['t']
        spotx = []
        spoty = []
        fills = []
        t0 = t[0]
        for i in range(1,len(t)):
            if np.isnan(t[i]):
                continue
            if abs(t[i]-t0) >= 1:
                spotx.append(x[i])
                spoty.append(y[i])
                if t[i] > t0:
                    fills.append('k')
                else:
                    fills.append('w')
                t0 = t[i]
        scat = pg.ScatterPlotItem(x=spotx, y=spoty, pen=self.pen, brush=fills, size=4)
        curve = pg.PlotDataItem(x=x, y=y, pen=self.pen)
        return curve, scat
        
        
        
def zeroCross(x):
    cs = 1024
    for i in range(0, len(x), cs):
        c = x[i:i+cs]
        mask = c > 0
        if not mask.any():
            continue
        if mask[0]:
            ind = 0
        else:
            ind = np.argwhere(np.diff(mask) > 0)[0,0] + 1
        return ind+i
    return None
    
    
#prog = [
    #(0.0,   0.2),
    #(2.0,   0.0),
    #(8.0,  -0.2),
    #(12.0,  0.0),
    #(18.0,  0.2),
#]
#clocks = [
    #Clock(x0=0.6),
    #Clock(x0=1.0, prog=[[0,0.5]]),
    ##Clock(x0=4, v0=0.0, prog=prog),
    ##Clock(x0=5, v0=0.0, prog=prog),
    #Clock(x0=0, v0=0.6),
    #Clock(x0=0.4, v0=0.6),
#]

f = 0.3
prog = [
    (6.0,   f),
    (8.0,   0.0),
    (14.0,  -f),
    (18.0,  0.0),
    (24.0,  f),
    (26.0,  0),
]
prog2 = [(t, -f) for (t,f) in prog] 

clocks = collections.OrderedDict()
for x in range(-10,11):
    clocks[('fixed', x)] = Clock(x0=x, pen=(100,100,100))

clocks.update({    ## all clocks included in the simulation
    'accel': Clock(x0=0.0, prog=prog, pen='r'),      ## accelerated twin
    'opposite': Clock(x0=0.0, prog=prog2, pen='b'),  ## opposite of accelerated twin
    'matched': Clock(x0=1.0, prog=prog, pen=(255,200,0)),     ## Offset from accelerated twin
    'matched2': Clock(x0=-1.0, prog=prog, pen=(255,200,0)),     ## Offset from accelerated twin
    'tag': Clock(x0=4.802*2, prog=prog2, pen='g'),       ## tags accelerated twin at the end of his journey
})


dt = 0.01
dur = 32.0
nPts = int(dur/dt)

w = pg.GraphicsWindow()
p1 = w.addPlot()

## Run simulation from inertial reference
for cl in clocks.itervalues():
    cl.startSim(nPts)
    for i in xrange(nPts):
        cl.advance(dt)
for cl in clocks.itervalues():
    #p1.addItem(cl.getCurve(clocks[('fixed',0)]))
    l, s = cl.getCurve()
    p1.addItem(s)
    p1.addItem(l)
    
## show simulation results transformed to another clock
w.nextRow()
p2 = w.addPlot()
for cl in clocks.itervalues():
    l, s = cl.getCurve(clocks['accel'])
    p2.addItem(s)
    p2.addItem(l)



