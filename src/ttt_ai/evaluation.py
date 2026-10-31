from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from .agents import Agent
from .game import Action, GameConfig, GameResult, GameState, Player, SearchBudget, SearchStats

@dataclass(frozen=True)
class AgentEntry:
    name: str
    factory: Callable[[int], Agent]
    metadata: dict[str, object]
    budget: SearchBudget | None = None

@dataclass(frozen=True)
class EvaluationConfig:
    games_per_pair: int = 100
    seeds: tuple[int, ...] = (42, 43, 44)
    split: Literal['validation', 'test'] = 'validation'
    mode: Literal['default', 'fixed_time', 'fixed_simulations'] = 'default'
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    output_dir: Path = Path('outputs/evaluation')

@dataclass(frozen=True)
class MoveLog:
    state: GameState
    action: Action
    stats: SearchStats

@dataclass(frozen=True)
class MatchLog:
    match_id: str
    block_id: str
    game_config: GameConfig
    protocol: EvaluationConfig
    agent_x: str
    agent_o: str
    seed_x: int
    seed_o: int
    opening: tuple[Action, ...]
    agent_metadata: dict[str, object]
    moves: tuple[MoveLog, ...]
    result: GameResult

@dataclass(frozen=True)
class TacticalCase:
    case_id: str
    game_config: GameConfig
    starting_player: Player
    history: tuple[Action, ...]
    acceptable_actions: frozenset[Action]
    split: Literal['validation', 'test']
    rationale: str

def play_match(game_config: GameConfig, agent_x: AgentEntry, agent_o: AgentEntry, protocol: EvaluationConfig, *, match_id: str, block_id: str, seed_x: int, seed_o: int, opening: tuple[Action, ...]=()) -> MatchLog:
    raise NotImplementedError

def run_tournament(game_config: GameConfig, entries: tuple[AgentEntry, ...], config: EvaluationConfig, openings: tuple[tuple[Action, ...], ...]=((),), *, smoke_test: bool=False) -> tuple[MatchLog, ...]:
    raise NotImplementedError

def summarize_results(logs: tuple[MatchLog, ...], config: EvaluationConfig) -> dict[str, object]:
    raise NotImplementedError

def save_logs(logs: tuple[MatchLog, ...], output_dir: Path) -> None:
    raise NotImplementedError

def load_logs(directory: Path) -> tuple[MatchLog, ...]:
    raise NotImplementedError

def run_ablation(kind: Literal['alpha_beta', 'mcts_simulations', 'guided_mcts'], game_config: GameConfig, entries: tuple[AgentEntry, ...], config: EvaluationConfig, states: tuple[GameState, ...]=()) -> dict[str, object]:
    raise NotImplementedError

def evaluate_tactics(entries: tuple[AgentEntry, ...], cases: tuple[TacticalCase, ...], seeds: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    raise NotImplementedError
