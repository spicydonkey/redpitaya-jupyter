"""Red Pitaya readout module.
"""

from enum import Enum, IntEnum

class InputRange(Enum):
    """Enum for input ranges"""
    LV = 1.0
    HV = 20.0

class TriggerEdge(Enum):
    """Enum for trigger edge"""
    POS = "pos"
    NEG = "neg"

class InputChannel(IntEnum):
    """Enum for input channels"""
    CH1 = 0
    CH2 = 1
