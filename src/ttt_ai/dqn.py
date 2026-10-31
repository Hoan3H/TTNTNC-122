from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Literal
from .agents import Agent
from .game import Action, AgentDecision, Game, GameConfig, GameState, Player, SearchBudget
if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from .evaluation import AgentEntry

@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 10000
    replay_warmup: int = 1000
    target_update_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    checkpoint_every: int = 100
    validation_every: int = 100
    validation_games: int = 20
    validation_seeds: tuple[int, ...] = (1001, 1002)
    transition_mode: Literal['same_player_next_turn'] = 'same_player_next_turn'
    encoding_version: str = 'two_channels_v1'

@dataclass(frozen=True)
class Transition:
    state: GameState
    action: Action
    reward: float
    next_state: GameState
    done: bool
    next_legal_actions: tuple[Action, ...]
    learner_player: Player

def encode_state(state: GameState, perspective: Player, config: GameConfig) -> Tensor:
    raise NotImplementedError

def build_network(game_config: GameConfig, device: str='cpu') -> nn.Module:
    raise NotImplementedError

def compute_td_targets(batch: tuple[Transition, ...], next_q_values: tuple[tuple[float, ...], ...], gamma: float) -> tuple[float, ...]:
    raise NotImplementedError

class DQNAgent:
    game_config: GameConfig
    config: DQNConfig
    rng: Random
    seed: int
    device: str
    model: nn.Module
    target_model: nn.Module | None
    optimizer: Optimizer | None
    replay: deque[Transition] | None
    completed_episodes: int
    global_step: int
    optimizer_steps: int
    metadata: dict[str, object]

    def __init__(self, game_config: GameConfig, config: DQNConfig | None=None, seed: int=42, device: str='cpu') -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None, *, epsilon: float=0.0) -> AgentDecision:
        raise NotImplementedError

    def predict_q_values(self, state: GameState) -> tuple[float, ...]:
        raise NotImplementedError

    def _collect_transition(self, game: Game, action: Action, learner_player: Player, opponent: Agent) -> Transition:
        raise NotImplementedError

    def _learn_batch(self, batch: tuple[Transition, ...]) -> float:
        raise NotImplementedError

    def _play_episode(self, opponent: Agent, learner_player: Player) -> dict[str, object]:
        raise NotImplementedError

    def train(self, opponent: Agent, validation_opponents: tuple[AgentEntry, ...], episodes: int=1000, output_dir: Path=Path('outputs/train'), resume_from: Path | None=None) -> Path:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path, *, resume: bool=False) -> None:
        raise NotImplementedError
