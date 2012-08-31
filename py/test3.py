import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import user
import collections
import sys
C = 1.0 

def gamma(v):
    return (1.0 - v**2 / C**2) ** -0.5

class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, y0=0.0, m0=1.0, v0=0.0, t0=0.0, pen=None, brush=None, prog=None):
        if pen is None:
            pen = pg.intColor(Clock.nClocks, 12)
        if brush is None:
            brush = (0,0,150)
        Clock.nClocks += 1
        self.pen = pg.mkPen(pen)
        self.brush = pg.mkBrush(brush)
        self.y0 = y0
        self.x0 = x0
        self.v0 = v0
        self.m0 = m0
        self.t0 = t0
        self.prog = prog

    def init(self, nPts):
        ## Keep records of object from inertial frame as well as reference frame
        self.inertData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float)])
        self.refData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float)])
        
        ## Inertial frame variables
        self.x = self.x0
        self.v = self.v0
        self.m = self.m0
        self.t = 0.0       ## reference clock always starts at 0
        self.pt = self.t0      ## proper time starts at t0
        
        ## reference frame variables
        self.refx = None
        self.refv = None
        self.refm = None
        self.reft = None
        
        self.recordFrame(0)
        
    def recordFrame(self, i):
        self.inertData[i] = (self.x, self.t, self.v, self.pt, self.m)
        self.refData[i] = (self.refx, self.reft, self.refv, self.pt, self.refm)
        
    def force(self, t):
        if self.prog is None:
            return 0.0
        
        ret = 0.0
        for t1,f in self.prog:
            if t >= t1:
                ret = f
        return ret
        
    def acceleration(self, t=None):
        if t is None:
            t = self.pt
        return self.force(t) / self.m0
        
    def accelLimits(self):
        ## return the proper time values which bound the current acceleration command
        if self.prog is None:
            return -np.inf, np.inf
        t = self.pt
        ind = -1
        for i, v in enumerate(self.prog):
            t1,f = v
            if t >= t1:
                ind = i
        
        if ind == -1:
            return -np.inf, self.prog[0][0]
        elif ind == len(self.prog)-1:
            return self.prog[-1][0], np.inf
        else:
            return self.prog[ind][0], self.prog[ind+1][0]
        
        
    def getCurve(self, ref=False):
        
        if ref is False:
            data = self.inertData
        else:
            data = self.refData[1:]
            
        x = data['x']
        y = data['t']
        
        curve = pg.PlotCurveItem(x=x, y=y, pen=self.pen)
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
        
        step = 1.0
        #mod = self.data['pt'] % step
        #inds = np.argwhere(abs(mod[1:] - mod[:-1]) > step*0.9)
        inds = [0]
        pt = data['pt']
        for i in range(1,len(pt)):
            diff = pt[i] - pt[inds[-1]]
            if abs(diff) >= step:
                inds.append(i)
        inds = np.array(inds)
        
        #t = self.data['t'][inds]
        #x = self.data['x'][inds]   
        pts = []
        for i in inds:
            x = data['x'][i]
            y = data['t'][i]
            if i+1 < len(data):
                dpt = data['pt'][i+1]-data['pt'][i]
                dt = data['t'][i+1]-data['t'][i]
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
    
def hypTStep(dt, v0, x0, tau0, g):
    ## Hyperbolic step. 
    ## If an object has proper acceleration g and starts at position x0 with speed v0 and proper time tau0
    ## as seen from an inertial frame, then return the new v, x, tau after time dt has elapsed.
    if g == 0:
        return v0, x0 + v0*dt, tau0 + dt * (1. - v0**2 / C**2)**0.5
    c2 = C**2
    c4 = c2**2
    v02 = v0**2
    g2 = g**2
    
    #A = (c2 * v0) / (-(g**2) * (-1 + v02))**0.5
    #if g > 0:
        #t0 = A
    #else:
        #t0 = -A
    tinit = (c2 * v0) / (g * (1 - v02)**0.5)
    
    B = c2 * (1 + (g2 * (dt+tinit)**2 / c4))**0.5
    
    v1 = g * (dt+tinit) / B
    
    dtau = (np.arcsinh(g * (dt+tinit)) - np.arcsinh(g * tinit)) / g
    
    tau1 = tau0 + dtau
    
    ## note: we compute the updated hyperbolic position here, but 
    ## this is not much different from using x0 + v0*dt
    #x1 = x0 + v0*dt
    #x1 = (-c2 * (1. / (1. - v0**2))**0.5 + c2 * B) / g + x0
    x1 = x0 + (c2 / g) * ( B - 1. / (1.-v02)**0.5 )
    
    return v1, x1, tau1


def tStep(dt, v0, x0, tau0, g):
    ## Linear step.
    ## Probably not as accurate as hyperbolic step, but certainly much faster.
    gamma = (1. - v0**2 / C**2)**-0.5
    dtau = dt / gamma
    return v0 + dtau * g, x0 + v0*dt, tau0 + dtau

def tauStep(dtau, v0, x0, t0, g):
    ## linear step in proper time of clock.
    ## If an object has proper acceleration g and starts at position x0 with speed v0 at time t0
    ## as seen from an inertial frame, then return the new v, x, t after proper time dtau has elapsed.
    

    ## Compute how much t will change given a proper-time step of dtau
    gamma = (1. - v0**2 / C**2)**-0.5
    if g == 0:
        dt = dtau * gamma
    else:
        v0g = v0 * gamma
        dt = (np.sinh(dtau * g + np.arcsinh(v0g)) - v0g) / g
    
    #return v0 + dtau * g, x0 + v0*dt, t0 + dt
    v1, x1, t1 = hypTStep(dt, v0, x0, t0, g)
    return v1, x1, t0+dt
    
def hypIntersect(x0r, t0r, vr, x0, t0, v0, g):
    ## given a reference clock (seen from inertial frame) has rx, rt, and rv,
    ## and another clock starts at x0, t0, and v0, with acceleration g,
    ## compute the intersection time of the object clock's hyperbolic path with 
    ## the reference plane.
    
    ## I'm sure we can simplify this...
    
    if g == 0:   ## no acceleration, path is linear (and hyperbola is undefined)
        #(-t0r + t0 v0 vr - vr x0 + vr x0r)/(-1 + v0 vr)
        
        t = (-t0r + t0 *v0 *vr - vr *x0 + vr *x0r)/(-1 + v0 *vr)
        return t
    
    gamma = (1.0-v0**2)**-0.5
    sel = (1 if g>0 else 0) + (1 if vr<0 else 0)
    sel = sel%2
    if sel == 0:
        #(1/(g^2 (-1 + vr^2)))(-g^2 t0r + g gamma vr + g^2 t0 vr^2 - 
        #g gamma v0 vr^2 - g^2 vr x0 + 
        #g^2 vr x0r + \[Sqrt](g^2 vr^2 (1 + gamma^2 (v0 - vr)^2 - vr^2 + 
        #2 g gamma (v0 - vr) (-t0 + t0r + vr (x0 - x0r)) + 
        #g^2 (t0 - t0r + vr (-x0 + x0r))^2)))
        
        t = (1./(g**2 *(-1. + vr**2)))*(-g**2 *t0r + g *gamma *vr + g**2 *t0 *vr**2 - g *gamma *v0 *vr**2 - g**2 *vr *x0 + g**2 *vr *x0r + np.sqrt(g**2 *vr**2 *(1. + gamma**2 *(v0 - vr)**2 - vr**2 + 2 *g *gamma *(v0 - vr)* (-t0 + t0r + vr *(x0 - x0r)) + g**2 *(t0 - t0r + vr* (-x0 + x0r))**2)))
        
    else:
        
        #-(1/(g^2 (-1 + vr^2)))(g^2 t0r - g gamma vr - g^2 t0 vr^2 + 
        #g gamma v0 vr^2 + g^2 vr x0 - 
        #g^2 vr x0r + \[Sqrt](g^2 vr^2 (1 + gamma^2 (v0 - vr)^2 - vr^2 + 
        #2 g gamma (v0 - vr) (-t0 + t0r + vr (x0 - x0r)) + 
        #g^2 (t0 - t0r + vr (-x0 + x0r))^2)))
      
        t = -(1./(g**2 *(-1. + vr**2)))*(g**2 *t0r - g *gamma* vr - g**2 *t0 *vr**2 + g *gamma *v0 *vr**2 + g**2* vr* x0 - g**2 *vr *x0r + np.sqrt(g**2* vr**2 *(1. + gamma**2 *(v0 - vr)**2 - vr**2 + 2 *g *gamma *(v0 - vr) *(-t0 + t0r + vr *(x0 - x0r)) + g**2 *(t0 - t0r + vr *(-x0 + x0r))**2)))
    return t
    
    
    

def run(dt, nPts, clocks, ref):
    
    for cl in clocks.itervalues():
        cl.init(nPts)
    
    ## make sure reference clock is not present in the list of clocks--this will be handled separately.
    clocks = clocks.copy()
    for k,v in clocks.iteritems():
        if v is ref:
            del clocks[k]
            break
    
    ref.refx = 0
    ref.refv = 0
    ref.refm = ref.m0
    
    ## These are the set of proper times (in the reference frame) that will be simulated
    ptVals = np.linspace(ref.pt, ref.pt + dt*(nPts-1), nPts)
    
    for i in xrange(1,nPts):
        if i % 100 == 0:
            print ".",
            sys.stdout.flush()
            
        ## step reference clock ahead one time step in its proper time
        nextPt = ptVals[i]  ## this is where (when) we want to end up
        while True:
            tau1, tau2 = ref.accelLimits()
            dtau = min(nextPt-ref.pt, tau2-ref.pt)  ## do not step past the next command boundary
            g = ref.acceleration()
            v, x, t = tauStep(dtau, ref.v, ref.x, ref.t, g)
            ref.pt += dtau
            ref.v = v
            ref.x = x
            ref.t = t
            ref.reft = ref.pt
            if ref.pt >= nextPt:
                break
            #else:
                #print "Stepped to", tau2, "instead of", nextPt
        ref.recordFrame(i)
        
        ## determine plane visible to reference clock
        ## this plane goes through the point ref.x, ref.t and has slope = ref.v
        
        
        ## update all other clocks
        for cl in clocks.itervalues():
            while True:
                g = cl.acceleration()
                tau1, tau2 = cl.accelLimits()
                ##Given current position / speed of clock, determine where it will intersect reference plane
                #t1 = (ref.v * (cl.x - cl.v * cl.t) + (ref.t - ref.v * ref.x)) / (1. - cl.v)
                t1 = hypIntersect(ref.x, ref.t, ref.v, cl.x, cl.t, cl.v, g)
                dt1 = t1 - cl.t
                
                ## advance clock by correct time step
                v, x, tau = hypTStep(dt1, cl.v, cl.x, cl.pt, g)
                
                ## check to see whether we have gone past an acceleration command boundary.
                ## if so, we must instead advance the clock to the boundary and start again
                if tau < tau1:
                    dtau = tau1 - cl.pt
                    cl.v, cl.x, cl.t = tauStep(dtau, cl.v, cl.x, cl.t, g)
                    cl.pt = tau1-0.000001  
                    continue
                if tau > tau2:
                    dtau = tau2 - cl.pt
                    cl.v, cl.x, cl.t = tauStep(dtau, cl.v, cl.x, cl.t, g)
                    cl.pt = tau2
                    continue
                
                ## Otherwise, record the new values and exit the loop
                cl.v = v
                cl.x = x
                cl.pt = tau
                cl.t = t1
                cl.m = None
                break
            
            ## transform position into reference frame
            x = cl.x - ref.x
            t = cl.t - ref.t
            gamma = (1.0 - ref.v**2 / C**2) ** -0.5
            vg = -ref.v * gamma
            
            cl.refx = gamma * (x - ref.v * t)
            cl.reft = ref.pt  #  + gamma * (t - ref.v * x)   # this term belongs here, but it should always be equal to 0.
            cl.refv = (cl.v - ref.v) / (1.0 - cl.v * ref.v)
            cl.refm = None
            cl.recordFrame(i)
            
            
            
            ##gam = gamma(cl.v)  ## (1.0 - cl.v**2 / C**2)**-0.5
            ##cl.m = cl.m0 * gam
            
            ##beta = (cl.x * ra / C**2) + 1
                
            ##dpt = dt * beta / gam   ## change in proper time for this clock
            
            ##f = cl.force(cl.pt)
            
            ##a = f / cl.m - ra/beta
            ###a = f / cl.m0 - ra*gam/beta
            
            ##cl.t += dt
            ##cl.pt2 += dpt
            
            ##cl.v, cl.x, cl.pt = vStep(dt * beta, cl.v, cl.x, cl.pt, a) 
            ###cl.v, x = vStep(dpt, cl.v, cl.x, a) 
            ###cl.x += dt * beta * cl.v
            
            cl.recordFrame(i)
        
        t += dt
    
def plot(clocks, plot, ref=False):
    for cl in clocks.itervalues():
        c, p = cl.getCurve(ref=ref)
        plot.addItem(c)
        plot.addItem(p)






class Animation(pg.ItemGroup):
    def __init__(self, clocks, start, stop, ref=False):
        pg.ItemGroup.__init__(self)
        self.clocks = clocks
        self.ref = ref
        self.tStart = start
        self.tStop = stop
        self.t = 0
        self.dt = 16
        
        self.items = {}
        for name, cl in clocks.items():
            item = ClockItem(cl, ref)
            self.addItem(item)
            self.items[name] = item
            
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(self.dt)
        
    def step(self):
        self.t += self.dt * 0.001
        if self.t > self.tStop:
            self.t = self.tStart
            
        for i in self.items.values():
            i.stepTo(self.t)

class ClockItem(pg.ItemGroup):
    def __init__(self, clock, ref):
        pg.ItemGroup.__init__(self)
        self.size = 0.4
        self.item = QtGui.QGraphicsEllipseItem(QtCore.QRectF(0, 0, self.size, self.size))
        self.item.translate(-self.size*0.5, -self.size*0.5)
        self.item.setPen(clock.pen)
        self.item.setBrush(clock.brush)
        self.hand = QtGui.QGraphicsLineItem(0, 0, 0, self.size*0.5)
        self.hand.setPen(pg.mkPen('w'))
        self.hand.setZValue(10)
        self.addItem(self.hand)
        self.addItem(self.item)
 
        self.clock = clock
        self.ref = ref
        self.i = 1
        
    def stepTo(self, t):
        if self.ref:
            data = self.clock.refData
        else:
            data = self.clock.inertData
        while self.i < len(data)-1 and data['t'][self.i] < t:
            self.i += 1
        while self.i > 1 and data['t'][self.i-1] >= t:
            self.i -= 1
        
        self.setPos(data['x'][self.i], self.clock.y0)
        
        t = data['pt'][self.i]
        self.hand.setRotation(-0.25 * t * 360.)
        
        self.resetTransform()
        v = data['v'][self.i]
        gam = (1.0 - v**2)**0.5
        self.scale(gam, 1.0)
        
        
    
def showAnimation(clocks, refClock):
    win = pg.GraphicsWindow()
    p1 = win.addPlot()
    p2 = win.addPlot(row=1, col=0)
    p1.setAspectLocked(1)
    p2.setAspectLocked(1)
    win.show()
    win.anim1 = Animation(clocks, 0, 35, ref=False)
    win.anim2 = Animation(clocks, 0, 35, ref=True)
    p1.addItem(win.anim1)
    p2.addItem(win.anim2)
    p1.autoRange()
    p2.autoRange()
    

        

        
        
win = pg.GraphicsWindow()

#dt = 0.001
#dur = 35.0
#nPts = int(dur/dt)+1
##run(dt, nPts, clocks, clocks[('fixed', 0)])

#p1 = win.addPlot()
#plot(clocks, p1)

#print "Inertial reference analysis:"
#analyze()
#print "Distance traveled:", clocks['accel'].data['x'].max()



## Acceleration program: accelerate, wait, reverse, wait, and stop
#f = 0.3017  ## 5.0
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
    'accel': Clock(x0=0.0, y0=1, prog=prog, pen='r'),      ## accelerated twin
    'opposite': Clock(x0=0.0, y0=2, prog=prog2, pen='b'),  ## opposite of accelerated twin
    'matched': Clock(x0=1.0, y0=1, prog=prog, pen=(255,200,0)),     ## Offset from accelerated twin
    'matched2': Clock(x0=-1.0, y0=1, prog=prog, pen=(255,200,0)),     ## Offset from accelerated twin
    'tag': Clock(x0=10., y0=2, prog=prog2, pen='g'),       ## tags accelerated twin at the end of his journey
})


dt = 0.01
dur = 32.0
nPts = int(dur/dt)+1

run(dt, nPts, clocks, clocks['accel'])

p1 = win.addPlot()
plot(clocks, p1, ref=False)
win.nextRow()
p2 = win.addPlot()
plot(clocks, p2, ref=True)

#print "\nAccelerated reference analysis:"
#analyze()

print "===  Length contraction test:  ==="

c1 = clocks[('fixed', 0)]
c2 = clocks[('fixed', -1)]

d1 = c2.inertData['x'][0] - c1.inertData['x'][0]
print "Clocks are %f apart in rest frame" % d1

ref = clocks['accel']
ind = np.argwhere(ref.refData['pt'] > 12)[0,0]
d2 = c2.refData['x'][ind] - c1.refData['x'][ind]
print "Clocks are %f apart at max speed" % d2

speed = ref.inertData['v'][ind]
print "Max speed is %f" % speed
print "Expected contraction:", (1.0 - speed**2/C**2)**0.5
print "Measured contraction:", d2 / d1

print "\n===  Time dilation test:  ==="

ind = np.argwhere(ref.inertData['t'] > 32.)[0,0]
t1 = c1.inertData['t'][ind]
t2 = ref.inertData['pt'][ind]
print "Time difference in rest frame:", t1-t2

t1 = c1.refData['pt'][ind]
t2 = ref.refData['pt'][ind]
print "Time difference in accelerated frame:", t1-t2


showAnimation(clocks, 'accel')