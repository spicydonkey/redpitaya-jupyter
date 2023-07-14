"""Red Pitaya readout module.
"""

from enum import auto, Enum, IntEnum

class InputRange(Enum):
    """Enum for input ranges"""
    LV_1 = 1.0
    HV_20 = 20.0

class TriggerEdge(Enum):
    """Enum for trigger edge"""
    POS = "pos"
    NEG = "neg"

class InputChannel(IntEnum):
    """Enum for input channels"""
    CH1 = 0
    CH2 = 1
