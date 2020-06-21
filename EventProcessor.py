
from Utility import *

class EventProcessor():

    def __init__(self):
        self.weightCorrector = None
        self.particleSelector = None

        self.event = None
        self.out = DotDict()
        self.internal = DotDict()


    def set_event(self, event):
        self.event = event
        
    def clear_event(self):
        self.event = None
        self.out.clear()
        self.internal.clear()


