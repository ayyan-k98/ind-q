"""
Data structures for MARL Coverage System
Contains all dataclasses used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from collections import defaultdict
import networkx as nx


@dataclass
class RobotState:
    """Represents the state of a single robot."""
    position: Tuple[int, int]
    orientation: float  # in radians


@dataclass
class WorldState:
    """Represents the global state of the world."""
    graph: nx.Graph  # Should only contain traversable nodes
    robots: List[RobotState]


@dataclass
class CoverageMetrics:
    """Class to track coverage performance metrics."""
    total_coverage: List[float] = field(default_factory=list)
    coverage_rate: List[float] = field(default_factory=list)
    redundant_coverage: List[float] = field(default_factory=list)  # Note: Redundant coverage not currently calculated
    agent_rewards: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    agent_losses: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    q_values: Dict[int, List[List[float]]] = field(default_factory=lambda: defaultdict(list))  # Stores history of Q-value arrays per agent
    planning_times: List[float] = field(default_factory=list)  # Note: Planning time not explicitly tracked
    communication_events: List[int] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)  # Tracks step execution time
    epsilon_values: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class CommunicationEvent:
    """Represents a communication event between agents."""
    sender_id: int
    receiver_id: int
    data_type: str  # 'coverage', 'position', 'map_exchange', etc.
    timestamp: float


@dataclass
class Room:
    """Helper class for room generation in map creation."""
    r1: int
    c1: int
    height: int
    width: int

    @property
    def r2(self) -> int:
        return self.r1 + self.height

    @property
    def c2(self) -> int:
        return self.c1 + self.width

    @property
    def center(self) -> Tuple[int, int]:
        return (self.r1 + self.height // 2, self.c1 + self.width // 2)

    def intersects(self, other: 'Room') -> bool:
        """Check if this room intersects with another room."""
        return (self.r1 < other.r2 and self.r2 > other.r1 and
                self.c1 < other.c2 and self.c2 > other.c1)
