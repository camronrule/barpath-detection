from enum import Enum

""" The big three lifts - squat, bench, and deadlift - can be classified into two main phases,
the eccentric (the down motion, or the stretch) and the concentric (the up motion, or the squeeze).
For the purposes of velocity based training, one is usually interested in the concentric, because this is 
where the sticking point is, which is where you probably fail the lift. 

Internally, the barbell is in one of the below phases.
    RACKED     -> Barbell is in the rack, not moving
    UNRACKING  -> The lifter is moving the barbell out of the rack, towards the top 
                  position of the eccentric
    TOP        -> The lifter is motionless, about to start the eccentric
    ECCENTRIC  -> The stretch
    BOTTOM     -> For the squat, the hole. For the bench, the chest.
    CONCENTRIC -> The squeeze (the slowest velocity point is likely the sticking point)
    RACKING    -> The lifter completed the concentric, and the lift is over.
"""


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
