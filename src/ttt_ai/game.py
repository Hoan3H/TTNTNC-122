from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Literal

Action = int
Player = Literal[-1, 1]
Cell = Literal[-1, 0, 1]
Board = tuple[tuple[Cell, ...], ...]

@dataclass(frozen=True)
class GameConfig:
    rows: int = 6
    cols: int = 7
    win_length: int = 4

@dataclass(frozen=True)
class GameResult:
    terminal: bool
    winner: Player | None
    is_draw: bool

@dataclass(frozen=True)
class GameState:
    board: Board
    to_play: Player
    move_count: int
    last_action: Action | None
    outcome: GameResult

@dataclass(frozen=True)
class SearchBudget:
    max_depth: int | None = None
    max_simulations: int | None = None
    time_limit_seconds: float | None = None

@dataclass(frozen=True)
class SearchStats:
    elapsed_seconds: float
    nodes_visited: int | None = None
    simulations: int | None = None
    completed_depth: int | None = None
    network_calls: int | None = None

@dataclass(frozen=True)
class AgentDecision:
    action: Action
    stats: SearchStats
    estimated_value: float | None = None

class Game:
    config: GameConfig
    _state: GameState

    def __init__(self, config: GameConfig | None = None) -> None:
        if config is None:
            config = GameConfig()
        if not isinstance(config, GameConfig):
            raise ValueError("config must be a GameConfig.")
        dimensions = (config.rows, config.cols, config.win_length)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Rows, columns and win length must be positive integers.")
        if config.win_length > max(config.rows, config.cols):
            raise ValueError("Win length exceeds every board direction.")
        self.config = config
        self.reset()

    def reset(self, starting_player: Player = 1) -> GameState:
        if type(starting_player) is not int or starting_player not in (-1, 1):
            raise ValueError("Starting player must be -1 or 1.")
        board = tuple(
            tuple(0 for _ in range(self.config.cols))
            for _ in range(self.config.rows)
        )
        self._state = GameState(
            board=board,
            to_play=starting_player,
            move_count=0,
            last_action=None,
            outcome=GameResult(terminal=False, winner=None, is_draw=False),
        )
        return self._state

    def snapshot(self) -> GameState:
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        if self.is_terminal():
            return ()
        return tuple(
            row * self.config.cols + col
            for row, cells in enumerate(self._state.board)
            for col, cell in enumerate(cells)
            if cell == 0
        )

    def step(self, action: Action) -> GameState:
        state = self._state
        if state.outcome.terminal:
            raise ValueError("Cannot move after the game has ended.")
        if type(action) is not int:
            raise ValueError("Action must be an integer.")
        total_cells = self.config.rows * self.config.cols
        if not 0 <= action < total_cells:
            raise ValueError("Action is outside the board.")
        row, col = divmod(action, self.config.cols)
        if state.board[row][col] != 0:
            raise ValueError("The selected cell is occupied.")

        new_row = state.board[row][:col] + (state.to_play,) + state.board[row][col + 1:]
        board = state.board[:row] + (new_row,) + state.board[row + 1:]
        move_count = state.move_count + 1
        won = self._check_win(action, state.to_play, board=board)
        outcome = GameResult(
            terminal=won or move_count == total_cells,
            winner=state.to_play if won else None,
            is_draw=not won and move_count == total_cells,
        )
        new_state = GameState(
            board=board,
            to_play=-state.to_play,
            move_count=move_count,
            last_action=action,
            outcome=outcome,
        )
        self._state = new_state
        return new_state

    def is_terminal(self) -> bool:
        return self._state.outcome.terminal

    def result(self) -> GameResult:
        return self._state.outcome

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        width = len(str(max(self.config.rows, self.config.cols)))
        lines = [
            " " * (width + 1)
            + " ".join(f"{col + 1:>{width}}" for col in range(self.config.cols))
        ]
        for row, cells in enumerate(self._state.board, start=1):
            lines.append(
                f"{row:>{width}} "
                + " ".join(f"{symbols[cell]:>{width}}" for cell in cells)
            )
        outcome = self.result()
        if outcome.winner is not None:
            status = f"Winner: {symbols[outcome.winner]}"
        elif outcome.is_draw:
            status = "Draw"
        else:
            status = f"Turn: {symbols[self._state.to_play]}"
        return "\n".join(lines + ["", status])

    def clone(self) -> Game:
        return copy(self)

    @classmethod
    def from_state(cls, config: GameConfig, state: GameState) -> Game:
        if not isinstance(config, GameConfig):
            raise ValueError("config must be a GameConfig.")
        game = cls(config)
        if not isinstance(state, GameState):
            raise ValueError("state must be a GameState.")
        board = state.board
        if not isinstance(board, tuple) or len(board) != config.rows:
            raise ValueError("Board must be a tuple with the configured number of rows.")
        if any(not isinstance(row, tuple) or len(row) != config.cols for row in board):
            raise ValueError("Each board row must be a tuple of the configured length.")
        if any(type(cell) is not int or cell not in (-1, 0, 1) for row in board for cell in row):
            raise ValueError("Board cells must be integers -1, 0 or 1.")
        if type(state.to_play) is not int or state.to_play not in (-1, 1):
            raise ValueError("Player to move must be -1 or 1.")
        if type(state.move_count) is not int:
            raise ValueError("Move count must be an integer.")
        move_count = sum(cell != 0 for row in board for cell in row)
        if state.move_count != move_count:
            raise ValueError("Move count does not match the occupied cells.")
        starting_player = state.to_play if move_count % 2 == 0 else -state.to_play
        starting_count = sum(cell == starting_player for row in board for cell in row)
        other_count = sum(cell == -starting_player for row in board for cell in row)
        if starting_count != (move_count + 1) // 2 or other_count != move_count // 2:
            raise ValueError("Piece counts are inconsistent with the player to move.")

        last_action = state.last_action
        if move_count == 0:
            if last_action is not None:
                raise ValueError("An empty board cannot have a last action.")
        else:
            if type(last_action) is not int or not 0 <= last_action < config.rows * config.cols:
                raise ValueError("Last action must identify a board cell.")
            last_row, last_col = divmod(last_action, config.cols)
            if board[last_row][last_col] != -state.to_play:
                raise ValueError("Last action must contain the previous player's piece.")

        outcome = state.outcome
        if not isinstance(outcome, GameResult):
            raise ValueError("Outcome must be a GameResult.")
        if type(outcome.terminal) is not bool or type(outcome.is_draw) is not bool:
            raise ValueError("Terminal and draw flags must be booleans.")
        if outcome.winner is not None and (
            type(outcome.winner) is not int or outcome.winner not in (-1, 1)
        ):
            raise ValueError("Winner must be -1, 1 or None.")

        winners = set()
        for row, cells in enumerate(board):
            for col, cell in enumerate(cells):
                if cell != 0 and game._check_win(row * config.cols + col, cell, board=board):
                    winners.add(cell)
        if len(winners) > 1:
            raise ValueError("Both players cannot have winning lines.")
        winner = next(iter(winners), None)
        full = move_count == config.rows * config.cols
        expected_outcome = GameResult(
            terminal=winner is not None or full,
            winner=winner,
            is_draw=winner is None and full,
        )
        if outcome != expected_outcome:
            raise ValueError("Outcome does not match the board.")

        if winner is not None:
            if last_action is None or winner != -state.to_play:
                raise ValueError("Winner must be the player who moved last.")
            if not game._check_win(last_action, winner, board=board):
                raise ValueError("The winning line must include the last action.")
            last_row, last_col = divmod(last_action, config.cols)
            previous_row = board[last_row][:last_col] + (0,) + board[last_row][last_col + 1:]
            previous_board = board[:last_row] + (previous_row,) + board[last_row + 1:]
            for row, cells in enumerate(previous_board):
                for col, cell in enumerate(cells):
                    if cell != 0 and game._check_win(
                        row * config.cols + col, cell, board=previous_board
                    ):
                        raise ValueError("The board contains moves after an earlier win.")

        game._state = state
        return game

    def _check_win(
        self, action: Action, player: Player, *, board: Board | None = None
    ) -> bool:
        if type(action) is not int or not 0 <= action < self.config.rows * self.config.cols:
            raise ValueError("Action must identify a board cell.")
        if type(player) is not int or player not in (-1, 1):
            raise ValueError("Player must be -1 or 1.")
        if board is None:
            board = self._state.board
        if not isinstance(board, tuple) or len(board) != self.config.rows:
            raise ValueError("Board shape does not match the configuration.")
        if any(not isinstance(row, tuple) or len(row) != self.config.cols for row in board):
            raise ValueError("Board shape does not match the configuration.")
        row, col = divmod(action, self.config.cols)
        if type(board[row][col]) is not int or board[row][col] != player:
            raise ValueError("The selected cell must contain the given player's piece.")
        for row_delta, col_delta in ((0, 1), (1, 0), (1, 1), (1, -1)):
            length = 1
            for sign in (-1, 1):
                next_row = row + sign * row_delta
                next_col = col + sign * col_delta
                while (
                    0 <= next_row < self.config.rows
                    and 0 <= next_col < self.config.cols
                    and board[next_row][next_col] == player
                ):
                    length += 1
                    next_row += sign * row_delta
                    next_col += sign * col_delta
            if length >= self.config.win_length:
                return True
        return False
