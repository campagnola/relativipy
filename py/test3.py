import pyqtgraph as pg
import numpy as np
import user
import collections

C = 1.0 

def gamma(v):
    return (1.0 - v**2 / C**2) ** -0.5

class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, m0=1.0, v0=0.0, pen=None, brush=None, prog=None):
        if pen is None:
            pen = pg.intColor(Clock.nClocks, 12)
        if brush is None:
            brush = (0,0,150)
        Clock.nClocks += 1
        self.pen = pg.mkPen(pen)
        self.brush = pg.mkBrush(brush)
        self.x0 = x0
        self.v0 = v0
        self.m0 = m0
        self.prog = prog

    def init(self, nPts):
        self.data = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float)])
        self.i = 0
        self.x = self.x0
        self.v = self.v0
        self.m = self.m0
        self.t = 0.0       ## reference clock always starts at 0
        self.pt = 0.0      ## proper time starts at t0
        self.recordFrame(0)
        
    def recordFrame(self, i):
        self.data[i] = (self.x, self.t, self.v, self.pt, self.m)
        
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
            gam = gamma(v)  # (1.0 - v**2 / c**2) ** -0.5
            x = gam * (self.data['x'] - v*self.data['t'] - ref.x0)
            y = gam * (self.data['t'] - v*self.data['x']/c**2)
            
        curve = pg.PlotCurveItem(x=x, y=y, pen=self.pen)
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
        
        step = 0.5
        #mod = self.data['pt'] % step
        #inds = np.argwhere(abs(mod[1:] - mod[:-1]) > step*0.9)
        inds = [0]
        pt = self.data['pt']
        for i in range(1,len(pt)):
            diff = pt[i] - pt[inds[-1]]
            if abs(diff) > step:
                inds.append(i)
        inds = np.array(inds)
        
        #t = self.data['t'][inds]
        #x = self.data['x'][inds]   
        pts = []
        for i in inds:
            x = self.data['x'][i]
            y = self.data['t'][i]
            if i+1 < len(self.data):
                dpt = self.data['pt'][i+1]-self.data['pt'][i]
                dt = self.data['t'][i+1]-self.data['t'][i]
            else:
                dpt = 1
                
            if dpt > 0:
                c = pg.mkBrush((0,0,0))
            else:
                c = pg.mkBrush((200,200,200))
            pts.append({'pos': (x, y), 'brush': c})
            
        points = pg.ScatterPlotItem(pts, pen=self.pen, size=5)
        
        return curve, points
        
        
#def hyperbolicTimestep(dt, x0, v0, a):
    ### Given a clock starts at x0 and v0 and accelerates a for dt, return x1 and v1
    #c2 = C**2
    #c4 = c2*c2
    #c6 = c4*c2
    #a2 = a**2
    
    #if v0 == 0:
        #A = (c2 * dt**2 + c4/a2)**0.5
        #x1 = x0 + A - (c4/a2)**0.5
        #v1 = (c2 * dt) / A
    #else:
        #v02 = v0**2
        #acmv2 = a2 * (c2 - v02)
        #A = c2 * v02   /   (c2 * v02 * acmv2)**0.5
        #if v0 < 0:
            #A *= -1
        #B = ((c4/a2) + c2 * (dt + A)**2)**0.5
        #print A, B
        #x1 = x0 - (c6 / acmv2)**0.5 + B
        #v1 = (c2 * (dt + A)) / B
    
    #return x1,v1
    
def vStep(dt, v0, x0, g):
    if g == 0:
        return v0, x0 + v0*dt
    c2 = C**2
    c4 = c2**2
    v02 = v0**2
    g2 = g**2
    A = (c2 * v0) / (-(g**2) * (-1 + v02))**0.5
    
    if g > 0:
        t0 = A
    else:
        t0 = -A
    
    B = c2 * (1 + (g2 * (dt+t0)**2 / c4))**0.5
    
    v1 = g * (dt+t0) / B
    
    ## note: we compute the updated hyperbolic position here, but 
    ## this is not much different from using x0 + v0*dt
    #x1 = x0 + v0*dt
    x1 = (-c2 * (1. / (1. - v0**2))**0.5 + c2 * B) / g + x0
    
    return v1, x1





def run(dt, nPts, clocks, ref):
    for cl in clocks.itervalues():
        cl.init(nPts)
    t = 0.0
    
    for i in xrange(1, nPts):
        
        ## force, acceleration, and position of reference clock
        rf = ref.force(t)
        ra = rf / ref.m0
        rx = ref.x
        
        
        ## update all clocks
        for cl in clocks.itervalues():
            gam = gamma(cl.v)  ## (1.0 - cl.v**2 / C**2)**-0.5
            cl.m = cl.m0 * gam
            
            beta = (cl.x * ra / C**2) + 1
                
            dpt = dt * beta / gam   ## change in proper time for this clock
            
            f = cl.force(cl.pt)
            
            a = f / cl.m - ra/beta
            #a = f / cl.m0 - ra*gam/beta
            
            cl.t += dt
            cl.pt += dpt
            
            cl.v, cl.x = vStep(dt * beta, cl.v, cl.x, a) 
            #cl.v, x = vStep(dpt, cl.v, cl.x, a) 
            #cl.x += dt * beta * cl.v
            
            cl.recordFrame(i)
        
        t += dt
    
def plot(clocks, plot):
    for cl in clocks.itervalues():
        c, p = cl.getCurve()
        plot.addItem(c)
        plot.addItem(p)







## Acceleration program: accelerate, wait, reverse, wait, and stop
#f = 0.3017  ## 5.0
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


def analyze():

    fixed0 = clocks[('fixed',0)]
    fixed1 = clocks[('fixed',-1)]
    l0 = fixed1.data['x'][0] - fixed0.data['x'][0]
    t1 = 5/dt
    l1 = fixed1.data['x'][t1] - fixed0.data['x'][t1]
    v = fixed1.data['v'][t1]
    dpt = (fixed1.data['pt'][t1]-fixed1.data['pt'][t1-1]) / dt
    print "  Fixed clocks:"
    print "    velocity:", v
    print "    length contraction:", l1/l0
    print "    time contraction:", dpt
    print "    expected contraction:", 1.0/gamma(v)

    print "  Twin:"
    print "    time difference:", fixed0.data['pt'][-1] - clocks['accel'].data['pt'][-1]
    
    
    
    
win = pg.GraphicsWindow()

dt = 0.001
dur = 35.0
nPts = int(dur/dt)+1
run(dt, nPts, clocks, clocks[('fixed', 0)])

p1 = win.addPlot()
plot(clocks, p1)

print "Inertial reference analysis:"
analyze()
print "Distance traveled:", clocks['accel'].data['x'].max()

win.nextRow()
p2 = win.addPlot()
dur = 25.0
nPts = int(dur/dt)+1

run(dt, nPts, clocks, clocks['accel'])
plot(clocks, p2)

print "\nAccelerated reference analysis:"
analyze()