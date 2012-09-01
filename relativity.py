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
            dict(name='Recalculate Worldlines', type='action'),
            dict(name='Animate', type='bool', value=True),
            self.objectGroup,
            ])
            
        self.tree.setParameters(self.params, showTop=False)
        
        
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


class ObjectGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Objects", addText="Add New..", addList=['Clock', 'Grid'])
        
    def addNew(self, typ):
        if typ == 'Clock':
            self.addChild(ClockParam())
        

class ClockParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Clock", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Initial Position', type='float', value=0.0, step=0.1),
            #dict(name='V0', type='float', value=0.0, step=0.1),
            AccelerationGroup(),
            
            dict(name='Rest Mass', type='float', value=1.0, step=0.1, limits=[1e-9, None]),
            dict(name='Color', type='color', value=(200,200,255)),
            dict(name='Show Clock', type='bool', value=True),
            ])
    

class AccelerationGroup(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Acceleration", addText="Add Command..")
        
    def addNew(self):
        nextTime = 0.0
        if self.hasChildren():
            nextTime = self.children()[-1]['Proper Time'] + 1
        self.addChild(Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True, children=[
            dict(name='Proper Time', type='float', value=nextTime),
            dict(name='Acceleration', type='float', value=0.0, step=0.1),
            ]))
    
if __name__ == '__main__':
    pg.mkQApp()
    win = RelativityGUI()
    win.setWindowTitle("Relativity!")
    win.show()

