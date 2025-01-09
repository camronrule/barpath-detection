from .BarbellPhase import BarbellPhase
from .strategy import PhaseDetectionStrategy
from .BarbellData import BarbellData


class DeadliftPhaseDetection(PhaseDetectionStrategy):

    def detect_phase(self, phase: BarbellPhase, BBD: BarbellData) -> BarbellPhase:

        v_x = BBD.get_v_x()
        v_y = BBD.get_v_y()

        match (phase):

            case BarbellPhase.RACKED:
                # we have x or y movement and rep has not been done yet
                # (unracking only happens before the reps start) -> unracking
                if ((v_y < -0.05) or (abs(v_x) > 0.05)) and not BBD.REP_COMPLETED:
                    phase = BarbellPhase.UNRACKING
                # check to see if video starts unracked already, therefore we skip
                # if abs(v_x < 0.01) and v_y > 0.0:
                    # phase = BarbellPhase.CONCENTRIC

            case BarbellPhase.UNRACKING:
                # no movement in x direction, and x position is
                # different from the x of the start of the rack position
                if (abs(v_x) < 0.005) and (abs(BBD.x_norms[-1] - BBD.phase_data[BarbellPhase.RACKED.name]["x"][-1]) > 0.005):
                    phase = BarbellPhase.TOP
                # elif we missed the 'TOP' -> if large y movement, go strait to concentric

            case BarbellPhase.TOP:  # if coming from a previous rep -> redraw line
                # if moving down in the y direction, and y_norm has changed since the beginning of TOP phase
                if (v_y > 0.04) and (abs(BBD.y_norms[-1] - BBD.phase_data[BarbellPhase.TOP.name]["y"][1]) > 0.01):
                    phase = BarbellPhase.ECCENTRIC
                # moving in x direction, and rep has been done
                # TODO check that x is moving closer to rack
                elif (abs(v_x) > 0.05) and BBD.REP_COMPLETED:
                    phase = BarbellPhase.RACKING

            case BarbellPhase.ECCENTRIC:
                # no longer moving down => in the hole
                if v_y < 0.03:
                    phase = BarbellPhase.BOTTOM

            case BarbellPhase.BOTTOM:
                # y velocity is negative when going up
                if v_y < -0.08:
                    phase = BarbellPhase.CONCENTRIC

            case BarbellPhase.CONCENTRIC:
                # y vel is negative when going up
                if v_y > -0.01:
                    phase = BarbellPhase.TOP
                    BBD.REP_COMPLETED = True

            case BarbellPhase.RACKING:
                if abs(v_x) < 0.03 and abs(v_y) < 0.03:
                    phase = BarbellPhase.RACKED

        return phase
