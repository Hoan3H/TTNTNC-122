from __future__ import annotations

from math import isfinite
from random import Random
from time import perf_counter
from typing import Protocol

from .game import (
    Action,
    AgentDecision,
    Game,
    GameConfig,
    GameState,
    Player,
    SearchBudget,
    SearchStats,
)


class Agent(Protocol):
    game_config: GameConfig
    rng: Random

    def select_action(
        self, state: GameState, budget: SearchBudget | None = None
    ) -> AgentDecision:
        raise NotImplementedError


class RandomAgent:
    game_config: GameConfig
    rng: Random

    def __init__(self, game_config: GameConfig, seed: int = 42) -> None:
        Game(game_config)
        if type(seed) is not int:
            raise ValueError("Seed must be an integer.")
        self.game_config = game_config
        self.rng = Random(seed)

    def select_action(
        self, state: GameState, budget: SearchBudget | None = None
    ) -> AgentDecision:
        started = perf_counter()
        if budget is not None and (
            not isinstance(budget, SearchBudget)
            or budget.max_depth is not None
            or budget.max_simulations is not None
            or budget.time_limit_seconds is not None
        ):
            raise ValueError("RandomAgent does not support a search budget.")
        game = Game.from_state(self.game_config, state)
        if game.is_terminal():
            raise ValueError("Cannot select an action for a terminal state.")
        action = self.rng.choice(game.legal_actions())
        return AgentDecision(
            action=action,
            stats=SearchStats(elapsed_seconds=perf_counter() - started),
        )


class RuleAgent:
    game_config: GameConfig
    rng: Random

    def __init__(self, game_config: GameConfig, seed: int = 42) -> None:
        Game(game_config)
        if type(seed) is not int:
            raise ValueError("Seed must be an integer.")
        self.game_config = game_config
        self.rng = Random(seed)

    def select_action(
        self, state: GameState, budget: SearchBudget | None = None
    ) -> AgentDecision:
        started = perf_counter()
        if budget is not None and (
            not isinstance(budget, SearchBudget)
            or budget.max_depth is not None
            or budget.max_simulations is not None
            or budget.time_limit_seconds is not None
        ):
            raise ValueError("RuleAgent does not support a search budget.")
        game = Game.from_state(self.game_config, state)
        if game.is_terminal():
            raise ValueError("Cannot select an action for a terminal state.")

        state = game.snapshot()
        player = state.to_play
        legal_actions = game.legal_actions()
        own_wins = winning_actions(state, self.game_config, player)
        if own_wins:
            action = own_wins[0]
        else:
            opponent_wins = winning_actions(state, self.game_config, -player)
            if opponent_wins:
                action = opponent_wins[0]
            else:
                action = legal_actions[0]
                best_value = float("-inf")
                for candidate in legal_actions:
                    child = game.clone()
                    next_state = child.step(candidate)
                    if next_state.outcome.terminal:
                        if next_state.outcome.winner == player:
                            value = 1.0
                        elif next_state.outcome.is_draw:
                            value = 0.0
                        else:
                            value = -1.0
                    else:
                        value = evaluate_position(
                            next_state, self.game_config, player
                        )
                    if value > best_value:
                        action = candidate
                        best_value = value

        child = game.clone()
        next_state = child.step(action)
        if next_state.outcome.terminal:
            if next_state.outcome.winner == player:
                estimated_value = 1.0
            elif next_state.outcome.is_draw:
                estimated_value = 0.0
            else:
                estimated_value = -1.0
        else:
            estimated_value = evaluate_position(
                next_state, self.game_config, player
            )
        return AgentDecision(
            action=action,
            stats=SearchStats(elapsed_seconds=perf_counter() - started),
            estimated_value=estimated_value,
        )


class MinimaxAgent:
    game_config: GameConfig
    rng: Random
    depth: int
    use_alpha_beta: bool
    _deadline: float | None
    _nodes_visited: int

    def __init__(
        self,
        game_config: GameConfig,
        depth: int = 3,
        use_alpha_beta: bool = True,
        seed: int = 42,
    ) -> None:
        Game(game_config)
        if type(depth) is not int or depth < 1:
            raise ValueError("Depth must be a positive integer.")
        if type(use_alpha_beta) is not bool:
            raise ValueError("use_alpha_beta must be a boolean.")
        if type(seed) is not int:
            raise ValueError("Seed must be an integer.")
        self.game_config = game_config
        self.rng = Random(seed)
        self.depth = depth
        self.use_alpha_beta = use_alpha_beta
        self._deadline = None
        self._nodes_visited = 0

    def select_action(
        self, state: GameState, budget: SearchBudget | None = None
    ) -> AgentDecision:
        started = perf_counter()
        if budget is not None and not isinstance(budget, SearchBudget):
            raise ValueError("budget must be a SearchBudget or None.")
        if budget is None:
            max_depth = self.depth
            time_limit_seconds = None
        else:
            if budget.max_simulations is not None:
                raise ValueError("MinimaxAgent does not support simulation limits.")
            max_depth = self.depth if budget.max_depth is None else budget.max_depth
            time_limit_seconds = budget.time_limit_seconds
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if time_limit_seconds is not None:
            if type(time_limit_seconds) not in (int, float):
                raise ValueError("time_limit_seconds must be finite and positive.")
            try:
                time_limit_seconds = float(time_limit_seconds)
            except OverflowError:
                raise ValueError(
                    "time_limit_seconds must be finite and positive."
                ) from None
            if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
                raise ValueError("time_limit_seconds must be finite and positive.")

        game = Game.from_state(self.game_config, state)
        if game.is_terminal():
            raise ValueError("Cannot select an action for a terminal state.")
        legal_actions = game.legal_actions()
        root_player = game.snapshot().to_play
        best_action = legal_actions[0]
        best_value: float | None = None
        completed_depth = 0
        self._nodes_visited = 0
        self._deadline = (
            started + time_limit_seconds
            if time_limit_seconds is not None
            else None
        )

        try:
            depths = (
                range(1, max_depth + 1)
                if time_limit_seconds is not None
                else (max_depth,)
            )
            for current_depth in depths:
                if (
                    self._deadline is not None
                    and perf_counter() >= self._deadline
                ):
                    break

                self._nodes_visited += 1
                iteration_action = legal_actions[0]
                iteration_value = float("-inf")
                alpha = float("-inf")
                beta = float("inf")
                try:
                    for action in legal_actions:
                        if (
                            self._deadline is not None
                            and perf_counter() >= self._deadline
                        ):
                            raise TimeoutError
                        child = game.clone()
                        child.step(action)
                        value = self._search(
                            child,
                            current_depth - 1,
                            root_player,
                            alpha,
                            beta,
                        )
                        if value > iteration_value:
                            iteration_action = action
                            iteration_value = value
                        if self.use_alpha_beta:
                            alpha = max(alpha, iteration_value)
                except TimeoutError:
                    break
                best_action = iteration_action
                best_value = iteration_value
                completed_depth = current_depth
        finally:
            self._deadline = None

        return AgentDecision(
            action=best_action,
            stats=SearchStats(
                elapsed_seconds=perf_counter() - started,
                nodes_visited=self._nodes_visited,
                completed_depth=completed_depth,
            ),
            estimated_value=best_value,
        )

    def _search(
        self,
        game: Game,
        depth: int,
        root_player: Player,
        alpha: float,
        beta: float,
    ) -> float:
        if not isinstance(game, Game) or game.config != self.game_config:
            raise ValueError("Search game must use this agent's configuration.")
        if type(depth) is not int or depth < 0:
            raise ValueError("Search depth must be a non-negative integer.")
        if type(root_player) is not int or root_player not in (-1, 1):
            raise ValueError("root_player must be -1 or 1.")
        if self._deadline is not None and perf_counter() >= self._deadline:
            raise TimeoutError
        self._nodes_visited += 1

        state = game.snapshot()
        if state.outcome.terminal:
            if state.outcome.winner == root_player:
                return 1.0
            if state.outcome.winner == -root_player:
                return -1.0
            return 0.0
        if depth == 0:
            return evaluate_position(state, self.game_config, root_player)

        legal_actions = game.legal_actions()
        if not legal_actions:
            return 0.0
        maximizing = state.to_play == root_player
        value = float("-inf") if maximizing else float("inf")
        for action in legal_actions:
            child = game.clone()
            child.step(action)
            child_value = self._search(
                child, depth - 1, root_player, alpha, beta
            )
            if maximizing:
                value = max(value, child_value)
                if self.use_alpha_beta:
                    alpha = max(alpha, value)
            else:
                value = min(value, child_value)
                if self.use_alpha_beta:
                    beta = min(beta, value)
            if self.use_alpha_beta and alpha >= beta:
                break
        return value


def evaluate_position(
    state: GameState, config: GameConfig, perspective: Player
) -> float:
    if type(perspective) is not int or perspective not in (-1, 1):
        raise ValueError("perspective must be -1 or 1.")
    game = Game.from_state(config, state)
    if game.is_terminal():
        raise ValueError("Cannot evaluate a terminal state.")

    board = state.board
    rows = config.rows
    cols = config.cols
    win_length = config.win_length
    total = 0
    for row_delta, col_delta in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(rows):
            for col in range(cols):
                end_row = row + (win_length - 1) * row_delta
                end_col = col + (win_length - 1) * col_delta
                if not (0 <= end_row < rows and 0 <= end_col < cols):
                    continue
                own_count = 0
                opponent_count = 0
                for offset in range(win_length):
                    cell = board[
                        row + offset * row_delta
                    ][col + offset * col_delta]
                    own_count += cell == perspective
                    opponent_count += cell == -perspective
                if own_count and not opponent_count:
                    total += 10 ** (own_count - 1)
                elif opponent_count and not own_count:
                    total -= 10 ** (opponent_count - 1)

    position = 0.0
    for row, cells in enumerate(board):
        row_distance = abs(2 * row - (rows - 1)) / max(rows - 1, 1)
        for col, cell in enumerate(cells):
            col_distance = abs(2 * col - (cols - 1)) / max(cols - 1, 1)
            center = 1 - (row_distance + col_distance) / 2
            position += cell * perspective * center
    position /= rows * cols

    raw = total + 0.1 * position
    return 0.99 * raw / (1 + abs(raw))


def winning_actions(
    state: GameState, config: GameConfig, player: Player
) -> tuple[Action, ...]:
    if type(player) is not int or player not in (-1, 1):
        raise ValueError("player must be -1 or 1.")
    game = Game.from_state(config, state)
    if game.is_terminal():
        return ()

    winning = []
    for action in game.legal_actions():
        row, col = divmod(action, config.cols)
        candidate_row = (
            state.board[row][:col]
            + (player,)
            + state.board[row][col + 1 :]
        )
        candidate_board = (
            state.board[:row]
            + (candidate_row,)
            + state.board[row + 1 :]
        )
        if game._check_win(action, player, board=candidate_board):
            winning.append(action)
    return tuple(winning)
