# default -> RACKED. 
# start drawing rep at TOP phase
from enum import Enum                        # BarbellPhase type

class BarbellPhase(Enum):
    RACKED = 1
    UNRACKING = 2
    TOP = 3
    ECCENTRIC = 4
    BOTTOM = 5
    CONCENTRIC = 6
    RACKING = 7