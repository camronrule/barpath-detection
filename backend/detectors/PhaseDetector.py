from .squat import SquatPhaseDetection
from .bench import BenchPhaseDetection
from .deadlift import DeadliftPhaseDetection
from .BarbellPhase import BarbellPhase
from .BarbellData import BarbellData


class PhaseDetector:
    """Select the phase detection strategy based on lift type

    Must be either: "Squat", "Bench", or "Deadlift"
    """

    def __init__(self, lift_type: str):
        if lift_type == "Squat":
            self.strategy = SquatPhaseDetection()
        elif lift_type == "Bench":
            self.strategy = BenchPhaseDetection()
        elif lift_type == "Deadlift":
            self.strategy = DeadliftPhaseDetection()
        else:
            raise ValueError(f"Unknown lift type: {lift_type}")

    def detect(self, phase: BarbellPhase, BBD: BarbellData):  # BBT : BarbellTracker
        return self.strategy.detect_phase(phase, BBD)
