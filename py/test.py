import pyqtgraph as pg
import numpy as np
import collections
c = 1.0

## It appears that this test does not work because the form of the lorentz transformation
## only works for objects moving at constant speed which share x(t=0)=0 origin with
## the standard frame.


class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, v0=0.0, prog=None, pen=None, brush=None):
        if pen is None:
            pen = pg.intColor(Clock.nClocks, 12)
        if brush is None:
            brush = (0,0,150)
        Clock.nClocks += 1
        self.x0 = x0
        self.v0 = v0
        self.prog = prog
        self.pen = pen
        self.brush = brush

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
            x = gamma * (self.data['x'] - v*self.data['t'] - ref.data['x'] + v*ref.data['t'])
            y = gamma * (self.data['t'] - v*self.data['x']/c**2 + v*ref.data['x']/c**2)
            
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
            
        return pg.PlotCurveItem(x=x, y=y, pen=self.pen)
        
        
        

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
    (0.0,   f),
    (2.0,   0.0),
    (8.0,  -f),
    (12.0,  0.0),
    (18.0,  f),
    (20.0,  0),
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


dt = 0.001
dur = 35.0
nPts = int(dur/dt)

w = pg.GraphicsWindow()
p1 = w.addPlot()

for cl in clocks.itervalues():
    cl.startSim(nPts)
    for i in xrange(nPts):
        cl.advance(dt)
for cl in clocks.itervalues():
    p1.addItem(cl.getCurve(clocks[('fixed',0)]))
    
w.nextRow()
p2 = w.addPlot()
for cl in clocks.itervalues():
    p2.addItem(cl.getCurve(clocks['accel']))



