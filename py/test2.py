import pyqtgraph as pg
import numpy as np


class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, m0=1.0, v0=0.0, color=None, prog=None):
        if color is None:
            color = pg.intColor(Clock.nClocks, 6)
        Clock.nClocks += 1
        self.color = color
        self.x0 = x0
        self.v0 = v0
        self.m0 = 1.0
        self.prog = prog

    def startSim(self, nPts):
        self.data = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float), ('f', float)])
        self.i = 0
        self.x = self.x0
        self.v = self.v0
        self.m = self.m0
        self.t = 0.0       ## reference clock always starts at 0
        self.pt = 0.0      ## proper time starts at t0
        
    def advance(self, t, C=1.0):
        dt = t-self.t
        try:
            gamma = (1.0 - self.v**2 / C**2) ** -0.5
        except: 
            print self.v
        self.pt += dt / gamma
        self.m = self.m0 * gamma
        f = self.force(self.t)
        self.v += (f/self.m) * dt
        if self.v >= 1.0:
            self.v = 0.999999999999   ## cheating. Use better integration method.
        if self.v <= -1.0:
            self.v = -0.999999999999
        self.x += self.v * dt
        #self.t += dt
        self.t = t
        
        self.data[self.i] = (self.x, self.t, self.v, self.pt, self.m, f)
        self.i += 1
    
    def force(self, t):
        if self.prog is None:
            return 0.0
        
        ret = 0.0
        for t1,f in self.prog:
            if t >= t1:
                ret = f
        return ret
        
    def getCurve(self, ref=None):
        if ref is None:
            x = self.data['x']
            y = self.data['t']
        else:
            v = ref.data['v']
            gamma = (1.0 - v**2 / c**2) ** -0.5
            x = gamma * (self.data['x'] - v*self.data['t'] - ref.x0)
            y = gamma * (self.data['t'] - v*self.data['x']/c**2)
            
        curve = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(self.color))
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
        
        mod = self.data['pt'] % 0.1
        inds = np.argwhere(mod[1:] < mod[:-1])
        t = self.data['t'][inds]
        x = self.data['x'][inds]   
            
        points = pg.ScatterPlotItem(x=x, y=t, pen=pg.mkPen(self.color), brush=pg.mkBrush('b'), size=4)
        
        return curve, points
        
        
        

f = 0.8
prog = [
    (0.0,   f),
    (2.0,   0.0),
    (8.0,  -f),
    (12.0,  0.0),
    (18.0,  f),
]
clocks = [
    Clock(x0=0.6),
    Clock(x0=1.0, prog=[[0,0.5]]),
    Clock(x0=4, v0=0.0, prog=prog),
    Clock(x0=5, v0=0.0, prog=prog),
    #Clock(x0=0, v0=0.6),
    #Clock(x0=0.4, v0=0.6),
]
dt = 0.001
dur = 20.0
nPts = int(dur/dt)


p1 = pg.plot()

for cl in clocks:
    cl.startSim(nPts)
    t = 0.0
    for i in xrange(nPts):
        cl.advance(t)
        t += dt
    c, p = cl.getCurve()
    p1.addCurve(c)
    p1.addDataItem(p)
    
    
#p2 = pg.plot()
#for cl in clocks:
    #p2.addCurve(cl.getCurve(clocks[2]))



