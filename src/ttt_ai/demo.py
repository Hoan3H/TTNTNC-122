from __future__ import annotations
from .agents import Agent
from .game import GameConfig, GameResult, Player

def play_human_vs_agent(game_config: GameConfig, agent: Agent, human_player: Player=1) -> GameResult:
    raise NotImplementedError

def main(argv: list[str] | None=None) -> None:
    raise NotImplementedError
