from enum import Enum


class BarbellPhase(Enum):
    """Represents the phase of the barbell lift
    """
    RACKED = 1
    UNRACKING = 2
    TOP = 3
    ECCENTRIC = 4
    BOTTOM = 5
    CONCENTRIC = 6
    RACKING = 7
