import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import numpy as np
import user
import collections
import sys



class RelativityGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        
        self.setupGUI()
        
        self.objectGroup = ObjectGroupParam()
        
        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Load Preset..', type='list', values=[]),
            #dict(name='Unit System', type='list', values=['', 'MKS']),
            dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
            dict(name='Reference Frame', type='list', values=[]),
            dict(name='Animate', type='bool', value=True),
            dict(name='Recalculate Worldlines', type='action'),
            self.objectGroup,
            ])
            
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Recalculate Worldlines').sigActivated.connect(self.recalculate)
        
        
    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        
        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)
        
        self.splitter2 = QtGui.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.splitter2)
        
        self.worldlinePlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.worldlinePlots)
        
        self.animationPlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.animationPlots)
        
        self.inertWorldlinePlot = self.worldlinePlots.addPlot()
        self.refWorldlinePlot = self.worldlinePlots.addPlot()
        
        self.inertAnimationPlot = self.animationPlots.addPlot()
        self.refAnimationPlot = self.animationPlots.addPlot()
        self.inertAnimationPlot.setXLink(self.inertWorldlinePlot)
        self.refAnimationPlot.setXLink(self.refWorldlinePlot)

    def recalculate(self):
        clocks = collections.OrderedDict()
        for cl in self.params.param('Objects'):
            clocks.update(cl.buildClocks())
        sim = Simulation(clocks, ref=None, duration=self.params['Duration'])
        sim.run()
        sim.plot(self.inertWorldlinePlot)

class ObjectGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Objects", addText="Add New..", addList=['Clock', 'Grid'])
        
    def addNew(self, typ):
        if typ == 'Clock':
            self.addChild(ClockParam())
        

class ClockParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Clock", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Initial Position', type='float', value=0.0, step=0.1),
            #dict(name='V0', type='float', value=0.0, step=0.1),
            AccelerationGroup(),
            
            dict(name='Rest Mass', type='float', value=1.0, step=0.1, limits=[1e-9, None]),
            dict(name='Color', type='color', value=(200,200,255)),
            dict(name='Show Clock', type='bool', value=True),
            ])
        defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
            
    def buildClocks(self):
        x0 = self['Initial Position']
        color = self['Color']
        m = self['Rest Mass']
        prog = self.param('Acceleration').generate()
        c = Clock(x0=x0, m0=m, color=color, prog=prog)
        return {self.name(): c}

pTypes.registerParameterType('Clock', ClockParam)
    

class AccelerationGroup(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Acceleration", addText="Add Command..")
        defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
        
    def addNew(self):
        nextTime = 0.0
        if self.hasChildren():
            nextTime = self.children()[-1]['Proper Time'] + 1
        self.addChild(Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True, children=[
            dict(name='Proper Time', type='float', value=nextTime),
            dict(name='Acceleration', type='float', value=0.0, step=0.1),
            ]))
            
    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Acceleration']))
        return prog    
        
pTypes.registerParameterType('AccelerationGroup', AccelerationGroup)

            
class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, y0=0.0, m0=1.0, v0=0.0, t0=0.0, color=None, prog=None):
        Clock.nClocks += 1
        self.pen = pg.mkPen(color)
        self.brush = pg.mkBrush(color)
        self.y0 = y0
        self.x0 = x0
        self.v0 = v0
        self.m0 = m0
        self.t0 = t0
        self.prog = prog

    def init(self, nPts):
        ## Keep records of object from inertial frame as well as reference frame
        self.inertData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float), ('f', float)])
        self.refData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float), ('f', float)])
        
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
        f = self.force()
        self.inertData[i] = (self.x, self.t, self.v, self.pt, self.m, f)
        self.refData[i] = (self.refx, self.reft, self.refv, self.pt, self.refm, f)
        
    def force(self, t=None):
        if len(self.prog) == 0:
            return 0.0
        if t is None:
            t = self.pt
        
        ret = 0.0
        for t1,f in self.prog:
            if t >= t1:
                ret = f
        return ret
        
    def acceleration(self, t=None):
        return self.force(t) / self.m0
        
    def accelLimits(self):
        ## return the proper time values which bound the current acceleration command
        if len(self.prog) == 0:
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
            
        points = pg.ScatterPlotItem(pts, pen=self.pen, size=7)
        
        return curve, points


class Simulation:
    def __init__(self, clocks, ref, duration):
        self.clocks = clocks
        self.ref = ref
        self.duration = duration
        self.dt = 0.015
    
    @staticmethod
    def hypTStep(dt, v0, x0, tau0, g):
        ## Hyperbolic step. 
        ## If an object has proper acceleration g and starts at position x0 with speed v0 and proper time tau0
        ## as seen from an inertial frame, then return the new v, x, tau after time dt has elapsed.
        if g == 0:
            return v0, x0 + v0*dt, tau0 + dt * (1. - v0**2)**0.5
        v02 = v0**2
        g2 = g**2
        
        tinit = v0 / (g * (1 - v02)**0.5)
        
        B = (1 + (g2 * (dt+tinit)**2))**0.5
        
        v1 = g * (dt+tinit) / B
        
        dtau = (np.arcsinh(g * (dt+tinit)) - np.arcsinh(g * tinit)) / g
        
        tau1 = tau0 + dtau
        
        x1 = x0 + (1.0 / g) * ( B - 1. / (1.-v02)**0.5 )
        
        return v1, x1, tau1


    @staticmethod
    def tStep(dt, v0, x0, tau0, g):
        ## Linear step.
        ## Probably not as accurate as hyperbolic step, but certainly much faster.
        gamma = (1. - v0**2)**-0.5
        dtau = dt / gamma
        return v0 + dtau * g, x0 + v0*dt, tau0 + dtau

    @staticmethod
    def tauStep(dtau, v0, x0, t0, g):
        ## linear step in proper time of clock.
        ## If an object has proper acceleration g and starts at position x0 with speed v0 at time t0
        ## as seen from an inertial frame, then return the new v, x, t after proper time dtau has elapsed.
        

        ## Compute how much t will change given a proper-time step of dtau
        gamma = (1. - v0**2)**-0.5
        if g == 0:
            dt = dtau * gamma
        else:
            v0g = v0 * gamma
            dt = (np.sinh(dtau * g + np.arcsinh(v0g)) - v0g) / g
        
        #return v0 + dtau * g, x0 + v0*dt, t0 + dt
        v1, x1, t1 = Simulation.hypTStep(dt, v0, x0, t0, g)
        return v1, x1, t0+dt
        
    @staticmethod
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
        
    def run(self):
        nPts = int(self.duration/self.dt)+1
        for cl in self.clocks.itervalues():
            cl.init(nPts)
            
        if self.ref is None:
            self.runInertial(nPts)
        else:
            self.runReference(nPts)
        
    def runInertial(self, nPts):
        clocks = self.clocks
        dt = self.dt
        tVals = np.linspace(0, dt*(nPts-1), nPts)
        for cl in self.clocks.itervalues():
            for i in xrange(1,nPts):
                nextT = tVals[i]
                while True:
                    tau1, tau2 = cl.accelLimits()
                    x = cl.x
                    v = cl.v
                    tau = cl.pt
                    g = cl.acceleration()
                    
                    v1, x1, tau1 = self.hypTStep(dt, v, x, tau, g)
                    if tau1 > tau2:
                        dtau = tau2-tau
                        cl.v, cl.x, cl.t = self.tauStep(dtau, v, x, cl.t, g)
                        cl.pt = tau2
                    else:
                        cl.v, cl.x, cl.pt = v1, x1, tau1
                        cl.t += dt
                        
                    if cl.t >= nextT:
                        cl.recordFrame(i)
                        break
            
        
    def runReference(self, nPts):
        clocks = self.clocks
        ref = self.ref
        dt = self.dt
        dur = self.duration
        
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
                gamma = (1.0 - ref.v**2) ** -0.5
                vg = -ref.v * gamma
                
                cl.refx = gamma * (x - ref.v * t)
                cl.reft = ref.pt  #  + gamma * (t - ref.v * x)   # this term belongs here, but it should always be equal to 0.
                cl.refv = (cl.v - ref.v) / (1.0 - cl.v * ref.v)
                cl.refm = None
                cl.recordFrame(i)
                
            t += dt
        
    def plot(self, plot):
        plot.clear()
        for cl in self.clocks.itervalues():
            c, p = cl.getCurve()
            plot.addItem(c)
            plot.addItem(p)

        

if __name__ == '__main__':
    pg.mkQApp()
    win = RelativityGUI()
    win.setWindowTitle("Relativity!")
    win.show()
    win.resize(1100,700)
    
    state = {'name': 'Objects', 'addText': 'Add New..', 'type': None, 'children': [
        {'name': 'Clock', 'default': None, 'renamable': True, 'type': 'Clock', 'children': [
            {'name': 'Initial Position', 'value': 0.0, 'step': 0.1, 'type': 'float'}, 
            {'name': 'Acceleration', 'addText': 'Add Command..', 'type': 'AccelerationGroup', 'children': [
                {'name': 'Command0', 'renamable': True, 'type': None, 'children': [
                    {'name': 'Proper Time', 'value': 0.0, 'type': 'float'}, 
                    {'name': 'Acceleration', 'value': 0.0, 'step': 0.1, 'removable': False, 'type': 'float'}]}, 
                {'name': 'Command1', 'renamable': True, 'type': None, 'children': [
                    {'name': 'Proper Time', 'value': 1.0, 'type': 'float'}, 
                    {'name': 'Acceleration', 'value': 1.0, 'step': 0.1, 'removable': False, 'type': 'float'}]}, 
                {'name': 'Command2', 'renamable': True, 'type': None, 'children': [
                    {'name': 'Proper Time', 'value': 2.0, 'type': 'float'}, 
                    {'name': 'Acceleration', 'value': -1.0, 'step': 0.1, 'removable': False, 'type': 'float'}]},
                {'name': 'Command3', 'renamable': True, 'type': None, 'children': [
                    {'name': 'Proper Time', 'value': 3.0, 'type': 'float'}, 
                    {'name': 'Acceleration', 'value': 0.0, 'step': 0.1, 'removable': False, 'type': 'float'}]}
                ]}, 
            {'name': 'Rest Mass', 'limits': [1e-09, None], 'default': 1.0, 'value': 1.0, 'step': 0.1, 'removable': False, 'type': 'float'}, {'name': 'Color', 'default': (200, 200, 255), 'value': (200, 200, 255), 'type': 'color'}, {'name': 'Show Clock', 'default': True, 'value': True, 'type': 'bool'}]}]}
    
    
    win.params.param('Objects').restoreState(state)
