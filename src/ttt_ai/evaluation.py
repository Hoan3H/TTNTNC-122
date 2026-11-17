from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
from typing import Callable, Literal

from .agents import Agent
from .game import (
    Action,
    AgentDecision,
    Game,
    GameConfig,
    GameResult,
    GameState,
    Player,
    SearchBudget,
    SearchStats,
)


_LOG_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_MATCHES_NAME = "matches.jsonl"


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
    split: Literal["validation", "test"] = "validation"
    mode: Literal["default", "fixed_time", "fixed_simulations"] = "default"
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    output_dir: Path = Path("outputs/evaluation")


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
    split: Literal["validation", "test"]
    rationale: str


def play_match(
    game_config: GameConfig,
    agent_x: AgentEntry,
    agent_o: AgentEntry,
    protocol: EvaluationConfig,
    *,
    match_id: str,
    block_id: str,
    seed_x: int,
    seed_o: int,
    opening: tuple[Action, ...] = (),
) -> MatchLog:
    Game(game_config)
    _validate_protocol(protocol)
    _validate_agent_entry(agent_x)
    _validate_agent_entry(agent_o)
    _validate_identifier(match_id, "match_id")
    _validate_identifier(block_id, "block_id")
    if type(seed_x) is not int or type(seed_o) is not int:
        raise ValueError("Agent seeds must be integers.")
    if not isinstance(opening, tuple) or any(type(action) is not int for action in opening):
        raise ValueError("opening must be a tuple of integer actions.")

    game = Game(game_config)
    for action in opening:
        game.step(action)

    player_agents: dict[Player, Agent] = {
        1: _create_agent(agent_x, game_config, seed_x),
        -1: _create_agent(agent_o, game_config, seed_o),
    }
    if player_agents[1] is player_agents[-1]:
        raise ValueError("Agent factories must create independent instances.")

    entries = {1: agent_x, -1: agent_o}
    seeds = {1: seed_x, -1: seed_o}
    move_logs = []
    while not game.is_terminal():
        state = game.snapshot()
        player = state.to_play
        decision = player_agents[player].select_action(
            state, entries[player].budget
        )
        if not isinstance(decision, AgentDecision):
            raise TypeError("Agent.select_action must return an AgentDecision.")
        if (
            type(decision.action) is not int
            or decision.action not in game.legal_actions()
        ):
            raise ValueError(
                f"Agent {entries[player].name!r} selected an illegal action."
            )
        _validate_search_stats(decision.stats)
        game.step(decision.action)
        move_logs.append(
            MoveLog(state=state, action=decision.action, stats=decision.stats)
        )

    metadata = {
        "X": {
            "agent": dict(agent_x.metadata),
            "budget": asdict(agent_x.budget) if agent_x.budget is not None else None,
        },
        "O": {
            "agent": dict(agent_o.metadata),
            "budget": asdict(agent_o.budget) if agent_o.budget is not None else None,
        },
    }
    return MatchLog(
        match_id=match_id,
        block_id=block_id,
        game_config=game_config,
        protocol=protocol,
        agent_x=agent_x.name,
        agent_o=agent_o.name,
        seed_x=seed_x,
        seed_o=seed_o,
        opening=opening,
        agent_metadata=metadata,
        moves=tuple(move_logs),
        result=game.result(),
    )


def run_tournament(game_config: GameConfig, entries: tuple[AgentEntry, ...], config: EvaluationConfig, openings: tuple[tuple[Action, ...], ...]=((),), *, smoke_test: bool=False) -> tuple[MatchLog, ...]:
    raise NotImplementedError


def summarize_results(logs: tuple[MatchLog, ...], config: EvaluationConfig) -> dict[str, object]:
    raise NotImplementedError


def save_logs(logs: tuple[MatchLog, ...], output_dir: Path) -> None:
    if not isinstance(output_dir, Path):
        raise ValueError("output_dir must be a Path.")
    if not isinstance(logs, tuple) or any(not isinstance(log, MatchLog) for log in logs):
        raise ValueError("logs must be a tuple of MatchLog objects.")

    seen_ids = set()
    serialized_logs = []
    for log in logs:
        _validate_match_log(log)
        if log.match_id in seen_ids:
            raise ValueError(f"Duplicate match_id: {log.match_id!r}.")
        seen_ids.add(log.match_id)
        record = asdict(log)
        record["protocol"]["output_dir"] = str(log.protocol.output_dir)
        serialized_logs.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )

    manifest = {
        "schema_version": _LOG_SCHEMA_VERSION,
        "match_count": len(serialized_logs),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    manifest_text = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=2,
    )
    matches_text = "\n".join(serialized_logs)
    if matches_text:
        matches_text += "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Log output path is not a directory: {output_dir}")
    (output_dir / _MANIFEST_NAME).write_text(manifest_text + "\n", encoding="utf-8")
    (output_dir / _MATCHES_NAME).write_text(matches_text, encoding="utf-8")


def load_logs(directory: Path) -> tuple[MatchLog, ...]:
    if not isinstance(directory, Path):
        raise ValueError("directory must be a Path.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {directory}")

    manifest_path = directory / _MANIFEST_NAME
    matches_path = directory / _MATCHES_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid log manifest: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "match_count",
        "python_version",
        "platform",
    }:
        raise ValueError("Log manifest has an invalid shape.")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != _LOG_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported log schema version.")
    if type(manifest["match_count"]) is not int or manifest["match_count"] < 0:
        raise ValueError("Manifest match_count must be a non-negative integer.")
    if not isinstance(manifest["python_version"], str) or not isinstance(
        manifest["platform"], str
    ):
        raise ValueError("Manifest runtime metadata must be strings.")

    loaded_logs = []
    seen_ids = set()
    with matches_path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                raise ValueError(f"Empty match record at line {line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid match record at line {line_number}: {error}"
                ) from error
            log = _decode_match_log(record)
            _validate_match_log(log)
            if log.match_id in seen_ids:
                raise ValueError(f"Duplicate match_id: {log.match_id!r}.")
            seen_ids.add(log.match_id)
            loaded_logs.append(log)

    if len(loaded_logs) != manifest["match_count"]:
        raise ValueError("Manifest match_count does not match the match log.")
    return tuple(loaded_logs)


def run_ablation(kind: Literal['alpha_beta', 'mcts_simulations', 'guided_mcts'], game_config: GameConfig, entries: tuple[AgentEntry, ...], config: EvaluationConfig, states: tuple[GameState, ...]=()) -> dict[str, object]:
    raise NotImplementedError


def evaluate_tactics(entries: tuple[AgentEntry, ...], cases: tuple[TacticalCase, ...], seeds: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    raise NotImplementedError


def _validate_protocol(protocol: EvaluationConfig) -> None:
    if not isinstance(protocol, EvaluationConfig):
        raise ValueError("protocol must be an EvaluationConfig.")
    if type(protocol.games_per_pair) is not int or protocol.games_per_pair < 1:
        raise ValueError("games_per_pair must be a positive integer.")
    if (
        not isinstance(protocol.seeds, tuple)
        or not protocol.seeds
        or any(type(seed) is not int for seed in protocol.seeds)
    ):
        raise ValueError("seeds must be a non-empty tuple of integers.")
    if protocol.split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'.")
    if protocol.mode not in ("default", "fixed_time", "fixed_simulations"):
        raise ValueError("Unsupported evaluation mode.")
    if (
        type(protocol.confidence_level) not in (int, float)
        or not math.isfinite(protocol.confidence_level)
        or not 0 < protocol.confidence_level < 1
    ):
        raise ValueError("confidence_level must be finite and between 0 and 1.")
    if type(protocol.bootstrap_samples) is not int or protocol.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    if not isinstance(protocol.output_dir, Path):
        raise ValueError("protocol.output_dir must be a Path.")


def _validate_agent_entry(entry: AgentEntry) -> None:
    if not isinstance(entry, AgentEntry):
        raise ValueError("Agents must be AgentEntry objects.")
    _validate_identifier(entry.name, "agent name")
    if not callable(entry.factory):
        raise ValueError("Agent factory must be callable.")
    if not isinstance(entry.metadata, dict):
        raise ValueError("Agent metadata must be a dictionary.")
    if entry.budget is not None and not isinstance(entry.budget, SearchBudget):
        raise ValueError("Agent budget must be a SearchBudget or None.")


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _create_agent(entry: AgentEntry, config: GameConfig, seed: int) -> Agent:
    agent = entry.factory(seed)
    if not callable(getattr(agent, "select_action", None)):
        raise TypeError("Agent factory must return an object with select_action.")
    if getattr(agent, "game_config", config) != config:
        raise ValueError("Agent factory returned an agent for a different game.")
    return agent


def _validate_search_stats(stats: SearchStats) -> None:
    if not isinstance(stats, SearchStats):
        raise TypeError("AgentDecision.stats must be SearchStats.")
    if (
        type(stats.elapsed_seconds) not in (int, float)
        or not math.isfinite(stats.elapsed_seconds)
        or stats.elapsed_seconds < 0
    ):
        raise ValueError("elapsed_seconds must be finite and non-negative.")
    for name in (
        "nodes_visited",
        "simulations",
        "completed_depth",
        "network_calls",
    ):
        value = getattr(stats, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None.")


def _validate_match_log(log: MatchLog) -> None:
    if not isinstance(log, MatchLog):
        raise ValueError("Each log must be a MatchLog.")
    _validate_identifier(log.match_id, "match_id")
    _validate_identifier(log.block_id, "block_id")
    _validate_identifier(log.agent_x, "agent_x")
    _validate_identifier(log.agent_o, "agent_o")
    if type(log.seed_x) is not int or type(log.seed_o) is not int:
        raise ValueError("Match seeds must be integers.")
    if not isinstance(log.opening, tuple) or any(
        type(action) is not int for action in log.opening
    ):
        raise ValueError("opening must be a tuple of integer actions.")
    if not isinstance(log.agent_metadata, dict) or set(log.agent_metadata) != {
        "X",
        "O",
    }:
        raise ValueError("agent_metadata must contain exactly X and O entries.")
    if not isinstance(log.agent_metadata["X"], dict) or not isinstance(
        log.agent_metadata["O"], dict
    ):
        raise ValueError("Each player's agent metadata must be a dictionary.")
    _validate_protocol(log.protocol)
    game = Game(log.game_config)
    if not isinstance(log.result, GameResult):
        raise ValueError("Match result must be a GameResult.")
    if (
        type(log.result.terminal) is not bool
        or type(log.result.is_draw) is not bool
        or (
            log.result.winner is not None
            and (
                type(log.result.winner) is not int
                or log.result.winner not in (-1, 1)
            )
        )
    ):
        raise ValueError("Match result contains invalid values.")
    if not isinstance(log.moves, tuple) or any(
        not isinstance(move, MoveLog) for move in log.moves
    ):
        raise ValueError("moves must be a tuple of MoveLog objects.")

    for action in log.opening:
        game.step(action)
    for move in log.moves:
        if game.is_terminal():
            raise ValueError("Match log contains moves after a terminal state.")
        if not isinstance(move.state, GameState):
            raise ValueError("Each move must contain a GameState.")
        Game.from_state(log.game_config, move.state)
        if move.state != game.snapshot():
            raise ValueError("Move state does not match the replayed match.")
        if type(move.action) is not int or move.action not in game.legal_actions():
            raise ValueError("Match log contains an illegal action.")
        _validate_search_stats(move.stats)
        game.step(move.action)
    if not game.is_terminal() or game.result() != log.result:
        raise ValueError("Match log does not replay to its recorded final result.")


def _decode_match_log(record: object) -> MatchLog:
    if not isinstance(record, dict) or set(record) != {
        "match_id",
        "block_id",
        "game_config",
        "protocol",
        "agent_x",
        "agent_o",
        "seed_x",
        "seed_o",
        "opening",
        "agent_metadata",
        "moves",
        "result",
    }:
        raise ValueError("Match record has an invalid shape.")

    config_data = record["game_config"]
    if not isinstance(config_data, dict) or set(config_data) != {
        "rows",
        "cols",
        "win_length",
    }:
        raise ValueError("Match game_config has an invalid shape.")
    game_config = GameConfig(**config_data)
    Game(game_config)

    protocol_data = record["protocol"]
    if not isinstance(protocol_data, dict) or set(protocol_data) != {
        "games_per_pair",
        "seeds",
        "split",
        "mode",
        "confidence_level",
        "bootstrap_samples",
        "output_dir",
    }:
        raise ValueError("Match protocol has an invalid shape.")
    if not isinstance(protocol_data["seeds"], list):
        raise ValueError("Match protocol seeds must be a JSON array.")
    if not isinstance(protocol_data["output_dir"], str):
        raise ValueError("Match protocol output_dir must be a string.")
    protocol = EvaluationConfig(
        games_per_pair=protocol_data["games_per_pair"],
        seeds=tuple(protocol_data["seeds"]),
        split=protocol_data["split"],
        mode=protocol_data["mode"],
        confidence_level=protocol_data["confidence_level"],
        bootstrap_samples=protocol_data["bootstrap_samples"],
        output_dir=Path(protocol_data["output_dir"]),
    )
    _validate_protocol(protocol)

    result_data = record["result"]
    if not isinstance(result_data, dict) or set(result_data) != {
        "terminal",
        "winner",
        "is_draw",
    }:
        raise ValueError("Match result has an invalid shape.")
    result = GameResult(**result_data)

    metadata = record["agent_metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"X", "O"}:
        raise ValueError("Match agent_metadata must contain X and O.")
    if not isinstance(metadata["X"], dict) or not isinstance(metadata["O"], dict):
        raise ValueError("Player agent metadata must be an object.")

    opening_data = record["opening"]
    if not isinstance(opening_data, list):
        raise ValueError("Match opening must be a JSON array.")
    moves_data = record["moves"]
    if not isinstance(moves_data, list):
        raise ValueError("Match moves must be a JSON array.")

    moves = []
    for move_data in moves_data:
        if not isinstance(move_data, dict) or set(move_data) != {
            "state",
            "action",
            "stats",
        }:
            raise ValueError("Move record has an invalid shape.")
        state_data = move_data["state"]
        if not isinstance(state_data, dict) or set(state_data) != {
            "board",
            "to_play",
            "move_count",
            "last_action",
            "outcome",
        }:
            raise ValueError("Move state has an invalid shape.")
        board_data = state_data["board"]
        if not isinstance(board_data, list) or any(
            not isinstance(row, list) for row in board_data
        ):
            raise ValueError("Move board must be an array of row arrays.")
        outcome_data = state_data["outcome"]
        if not isinstance(outcome_data, dict) or set(outcome_data) != {
            "terminal",
            "winner",
            "is_draw",
        }:
            raise ValueError("Move outcome has an invalid shape.")
        state = GameState(
            board=tuple(tuple(row) for row in board_data),
            to_play=state_data["to_play"],
            move_count=state_data["move_count"],
            last_action=state_data["last_action"],
            outcome=GameResult(**outcome_data),
        )

        stats_data = move_data["stats"]
        if not isinstance(stats_data, dict) or set(stats_data) != {
            "elapsed_seconds",
            "nodes_visited",
            "simulations",
            "completed_depth",
            "network_calls",
        }:
            raise ValueError("Move stats have an invalid shape.")
        stats = SearchStats(**stats_data)
        moves.append(
            MoveLog(state=state, action=move_data["action"], stats=stats)
        )

    return MatchLog(
        match_id=record["match_id"],
        block_id=record["block_id"],
        game_config=game_config,
        protocol=protocol,
        agent_x=record["agent_x"],
        agent_o=record["agent_o"],
        seed_x=record["seed_x"],
        seed_o=record["seed_o"],
        opening=tuple(opening_data),
        agent_metadata=metadata,
        moves=tuple(moves),
        result=result,
    )
