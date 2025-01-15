from .BarbellPhase import BarbellPhase
from .strategy import PhaseDetectionStrategy
from .BarbellData import BarbellData


class DeadliftPhaseDetection(PhaseDetectionStrategy):
    """Detect phase in a deadlift

    Phases are:
        1. Bottom
        2. Concentric
        3. Top
        4. Eccentric
        5. Bottom
    """

    def detect_phase(self, phase: BarbellPhase, BBD: BarbellData) -> BarbellPhase:
        """Detect phase in a deadlift

        Args:
            phase (BarbellPhase): The phase in the previous frame
            BBD (BarbellData): The data for this barbell from the barbell tracker

        Returns:
            BarbellPhase: The predicted current phase of the deadlift
        """

        v_x = BBD.get_v_x()
        v_y = BBD.get_v_y()

        # 'RACKED' is default, but internally
        # a dl starts at 'BOTTOM'
        if phase == BarbellPhase.RACKED:
            phase = BarbellPhase.BOTTOM

        match (phase):

            case BarbellPhase.BOTTOM:
                if (abs(v_y) > 0.025):
                    phase = BarbellPhase.CONCENTRIC
                    BBD.increment_rep_num()  # rep_num starts at 0

            case BarbellPhase.CONCENTRIC:
                if (abs(v_y) < 0.03):  # TODO check for y diff from bottom
                    phase = BarbellPhase.TOP

            case BarbellPhase.TOP:
                if (abs(v_y) > 0.03):
                    phase = BarbellPhase.ECCENTRIC

            case BarbellPhase.ECCENTRIC:
                if (abs(v_y) < 0.03):
                    phase = BarbellPhase.BOTTOM

        return phase
