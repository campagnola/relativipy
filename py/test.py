import pyqtgraph as pg
import numpy as np

c = 1.0

class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, v0=0.0, color=None, prog=None):
        if color is None:
            color = pg.intColor(Clock.nClocks, 6)
        Clock.nClocks += 1
        self.color = color
        self.x0 = x0
        self.v0 = v0
        self.prog = prog

    def startSim(self, nPts):
        self.data = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float)])
        self.i = 0
        self.x = self.x0
        self.v = self.v0
        self.t = 0.0       ## reference clock always starts at 0
        self.pt = 0.0      ## proper time starts at t0
        
    def advance(self, dt):
        gamma = (1.0 - self.v**2 / c**2) ** -0.5
        g = self.acceleration()
        self.v += g * dt / gamma
        if self.v >= 1.0:
            self.v = 0.999999999999   ## cheating. Use better integration method.
        self.x += self.v * dt
        self.t += dt
        self.pt += dt * gamma
        
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
        if ref is None:
            x = self.data['x']
            y = self.data['t']
        else:
            v = ref.data['v']
            gamma = (1.0 - v**2 / c**2) ** -0.5
            x = gamma * (self.data['x'] - v*self.data['t'] - ref.x0)
            y = gamma * (self.data['t'] - v*self.data['x']/c**2)
            
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
            
        return pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(self.color))
        
        
        

prog = [
    (0.0,   0.2),
    (2.0,   0.0),
    (8.0,  -0.2),
    (12.0,  0.0),
    (18.0,  0.2),
]
clocks = [
    Clock(x0=0.6),
    Clock(x0=1.0, prog=[[0,0.5]]),
    #Clock(x0=4, v0=0.0, prog=prog),
    #Clock(x0=5, v0=0.0, prog=prog),
    Clock(x0=0, v0=0.6),
    Clock(x0=0.4, v0=0.6),
]
dt = 0.001
dur = 6.0
nPts = int(dur/dt)


p1 = pg.plot()

for cl in clocks:
    cl.startSim(nPts)
    for i in xrange(nPts):
        cl.advance(dt)
    p1.addCurve(cl.getCurve())
    
    
p2 = pg.plot()
for cl in clocks:
    p2.addCurve(cl.getCurve(clocks[2]))



