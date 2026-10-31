from __future__ import annotations
from dataclasses import dataclass, field
from random import Random
from .agents import Agent
from .game import Action, AgentDecision, GameConfig, GameState, Player, SearchBudget

@dataclass
class MCTSNode:
    state: GameState
    root_player: Player
    parent: MCTSNode | None = None
    action: Action | None = None
    children: dict[Action, MCTSNode] = field(default_factory=dict)
    untried: list[Action] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

class MCTSAgent:
    game_config: GameConfig
    rng: Random
    simulations: int
    exploration_constant: float
    time_limit_seconds: float | None
    guidance: Agent | None
    rollout_epsilon: float
    _deadline: float | None
    _network_calls: int
    _nodes_visited: int

    def __init__(self, game_config: GameConfig, simulations: int=100, exploration_constant: float=1.414, time_limit_seconds: float | None=None, seed: int=42) -> None:
        raise NotImplementedError

    def select_action(self, state: GameState, budget: SearchBudget | None=None) -> AgentDecision:
        raise NotImplementedError

    def _select_expand(self, root: MCTSNode) -> MCTSNode:
        raise NotImplementedError

    def _rollout(self, node: MCTSNode) -> float:
        raise NotImplementedError

    def _backpropagate(self, node: MCTSNode, root_value: float) -> None:
        raise NotImplementedError

    def _uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        raise NotImplementedError
