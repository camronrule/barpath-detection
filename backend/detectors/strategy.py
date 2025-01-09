from abc import ABC, abstractmethod


class PhaseDetectionStrategy(ABC):

    """Abstract base class for phase detection strategies.
    This class defines the interface that all phase detection strategies (S, B, and D) must implement.

    Methods:
        detect_phase(frames, **kwargs):
            Abstract method to detect the phase of a lift from the current frame and BB data.
    """

    @abstractmethod
    def detect_phase(self, frames, **kwargs):
        pass
