from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Literal
from .agents import Agent
from .dqn import DQNAgent, build_network, encode_state
from .game import Action, AgentDecision, GameConfig, GameResult, GameState, SearchBudget
from .mcts import MCTSAgent
if TYPE_CHECKING:
    from torch import nn

@dataclass(frozen=True)
class TeacherGame:
    game_id: str
    seed: int
    config: GameConfig
    teacher_metadata: dict[str, object]
    budget: SearchBudget | None
    samples: tuple[tuple[GameState, Action], ...]
    result: GameResult

def generate_dataset(game_config: GameConfig, teacher: Agent, games: int, seed: int, teacher_metadata: dict[str, object], budget: SearchBudget | None=None) -> tuple[TeacherGame, ...]:
    raise NotImplementedError

class SupervisedAgent:
    game_config: GameConfig
    rng: Random
    device: str
    model: nn.Module | None
    metadata: dict[str, object]

    def __init__(self, game_config: GameConfig, seed: int=42, device: str='cpu') -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

    def train(self, games: tuple[TeacherGame, ...], epochs: int=20, output_dir: Path=Path('outputs/policy')) -> Path:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

class GuidedMCTSAgent(MCTSAgent):
    guidance_kind: Literal['supervised_policy', 'dqn_q']

    def __init__(self, game_config: GameConfig, guidance: SupervisedAgent | DQNAgent, guidance_kind: Literal['supervised_policy', 'dqn_q'], rollout_epsilon: float=0.1, simulations: int=100, seed: int=42) -> None:
        raise NotImplementedError
