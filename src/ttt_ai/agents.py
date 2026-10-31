from __future__ import annotations
from random import Random
from typing import Protocol
from .game import Action, AgentDecision, Game, GameConfig, GameState, Player, SearchBudget

class Agent(Protocol):
    game_config: GameConfig
    rng: Random

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

class RandomAgent:
    game_config: GameConfig
    rng: Random

    def __init__(self, game_config: GameConfig, seed: int=42) -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

class RuleAgent:
    game_config: GameConfig
    rng: Random

    def __init__(self, game_config: GameConfig, seed: int=42) -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

class MinimaxAgent:
    game_config: GameConfig
    rng: Random
    depth: int
    use_alpha_beta: bool
    _deadline: float | None
    _nodes_visited: int

    def __init__(self, game_config: GameConfig, depth: int=3, use_alpha_beta: bool=True, seed: int=42) -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

    def _search(self, game: Game, depth: int, root_player: Player, alpha: float, beta: float) -> float:
        raise NotImplementedError

def evaluate_position(state: GameState, config: GameConfig, perspective: Player) -> float:
    raise NotImplementedError

def winning_actions(state: GameState, config: GameConfig, player: Player) -> tuple[Action, ...]:
    raise NotImplementedError
