from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from .evaluation import AgentEntry


_CHECKPOINT_SCHEMA_VERSION = 1


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
    transition_mode: Literal["same_player_next_turn"] = "same_player_next_turn"
    encoding_version: str = "two_channels_v1"


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
    if type(perspective) is not int or perspective not in (-1, 1):
        raise ValueError("perspective must be -1 or 1.")
    Game.from_state(config, state)
    import torch

    own_pieces = [
        [1.0 if cell == perspective else 0.0 for cell in row]
        for row in state.board
    ]
    opponent_pieces = [
        [1.0 if cell == -perspective else 0.0 for cell in row]
        for row in state.board
    ]
    return torch.tensor(
        [own_pieces, opponent_pieces], dtype=torch.float32
    ).unsqueeze(0)


def build_network(game_config: GameConfig, device: str = "cpu") -> nn.Module:
    Game(game_config)
    import torch
    from torch import nn

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty device string.")
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid torch device: {device!r}.") from error
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    if resolved_device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be CPU, CUDA or MPS.")

    model = nn.Sequential(
        nn.Conv2d(2, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * game_config.rows * game_config.cols,
                  game_config.rows * game_config.cols),
    )
    return model.to(resolved_device)


def compute_td_targets(
    batch: tuple[Transition, ...],
    next_q_values: tuple[tuple[float, ...], ...],
    gamma: float,
) -> tuple[float, ...]:
    if not isinstance(batch, tuple) or not batch:
        raise ValueError("batch must be a non-empty tuple of transitions.")
    if not isinstance(next_q_values, tuple) or len(next_q_values) != len(batch):
        raise ValueError("next_q_values must contain one row per transition.")
    if (
        type(gamma) not in (int, float)
        or not math.isfinite(gamma)
        or not 0 <= gamma <= 1
    ):
        raise ValueError("gamma must be finite and between 0 and 1.")

    targets = []
    for transition, q_values in zip(batch, next_q_values):
        if not isinstance(transition, Transition):
            raise ValueError("batch entries must be Transition objects.")
        if type(transition.reward) not in (int, float) or not math.isfinite(
            transition.reward
        ):
            raise ValueError("Transition reward must be finite.")
        if transition.reward not in (-1, 0, 1):
            raise ValueError("Transition reward must be -1, 0 or 1.")
        if type(transition.done) is not bool:
            raise ValueError("Transition done must be a boolean.")
        if not isinstance(transition.next_state, GameState):
            raise ValueError("Transition next_state must be a GameState.")
        if (
            type(transition.learner_player) is not int
            or transition.learner_player not in (-1, 1)
            or transition.state.to_play != transition.learner_player
            or transition.done != transition.next_state.outcome.terminal
        ):
            raise ValueError("Transition turn or terminal state is inconsistent.")
        rows = len(transition.next_state.board)
        if rows == 0 or not isinstance(transition.next_state.board, tuple):
            raise ValueError("Transition next_state must contain a board.")
        cols = len(transition.next_state.board[0])
        if cols == 0 or any(len(row) != cols for row in transition.next_state.board):
            raise ValueError("Transition next_state board must be rectangular.")
        action_count = rows * cols
        if (
            not isinstance(q_values, tuple)
            or len(q_values) != action_count
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in q_values
            )
        ):
            raise ValueError("Each Q-value row must be finite and match board size.")

        if transition.done:
            if transition.next_legal_actions != ():
                raise ValueError("Terminal transitions cannot have legal actions.")
            targets.append(float(transition.reward))
            continue
        if (
            not isinstance(transition.next_legal_actions, tuple)
            or not transition.next_legal_actions
            or any(
                type(action) is not int or not 0 <= action < action_count
                for action in transition.next_legal_actions
            )
        ):
            raise ValueError(
                "A non-terminal transition must contain valid legal actions."
            )
        expected_legal_actions = tuple(
            row * cols + col
            for row, cells in enumerate(transition.next_state.board)
            for col, cell in enumerate(cells)
            if cell == 0
        )
        if (
            transition.next_state.to_play != transition.learner_player
            or transition.next_legal_actions != expected_legal_actions
        ):
            raise ValueError(
                "Transition next state must be the learner's turn with matching legal actions."
            )
        targets.append(
            float(
                transition.reward
                + gamma
                * max(q_values[action] for action in transition.next_legal_actions)
            )
        )
    return tuple(targets)


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

    def __init__(
        self,
        game_config: GameConfig,
        config: DQNConfig | None = None,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        import torch

        Game(game_config)
        if config is None:
            config = DQNConfig()
        if not isinstance(config, DQNConfig):
            raise ValueError("config must be a DQNConfig.")
        if type(seed) is not int:
            raise ValueError("seed must be an integer.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty device string.")

        float_values = (
            ("gamma", config.gamma, 0.0, 1.0),
            ("learning_rate", config.learning_rate, 0.0, math.inf),
            ("epsilon_start", config.epsilon_start, 0.0, 1.0),
            ("epsilon_end", config.epsilon_end, 0.0, 1.0),
        )
        for name, value, lower, upper in float_values:
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < lower
                or value > upper
                or (name == "learning_rate" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and in its allowed range.")
        if config.epsilon_start < config.epsilon_end:
            raise ValueError("epsilon_start must not be less than epsilon_end.")
        for name in (
            "batch_size",
            "replay_capacity",
            "replay_warmup",
            "target_update_steps",
            "epsilon_decay_steps",
            "checkpoint_every",
            "validation_every",
            "validation_games",
        ):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if config.batch_size > config.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity.")
        if config.replay_warmup > config.replay_capacity:
            raise ValueError("replay_warmup cannot exceed replay_capacity.")
        if (
            not isinstance(config.validation_seeds, tuple)
            or not config.validation_seeds
            or any(type(value) is not int for value in config.validation_seeds)
        ):
            raise ValueError("validation_seeds must be a non-empty tuple of integers.")
        if config.transition_mode != "same_player_next_turn":
            raise ValueError("Unsupported transition_mode.")
        if config.encoding_version != "two_channels_v1":
            raise ValueError("Unsupported encoding_version.")

        try:
            resolved_device = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid torch device: {device!r}.") from error
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if resolved_device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        if resolved_device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be CPU, CUDA or MPS.")

        self.game_config = game_config
        self.config = config
        self.seed = seed
        self.device = str(resolved_device)
        self.rng = Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = build_network(game_config, self.device)
            self.target_model = build_network(game_config, self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.replay = deque(maxlen=config.replay_capacity)
        self.completed_episodes = 0
        self.global_step = 0
        self.optimizer_steps = 0
        self.metadata = {
            "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "encoding_version": config.encoding_version,
            "best_validation_score": None,
        }

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
        *,
        epsilon: float = 0.0,
    ) -> AgentDecision:
        started = perf_counter()
        if budget is not None and (
            not isinstance(budget, SearchBudget)
            or budget.max_depth is not None
            or budget.max_simulations is not None
            or budget.time_limit_seconds is not None
        ):
            raise ValueError("DQNAgent does not support a search budget.")
        if (
            type(epsilon) not in (int, float)
            or not math.isfinite(epsilon)
            or not 0 <= epsilon <= 1
        ):
            raise ValueError("epsilon must be finite and between 0 and 1.")
        game = Game.from_state(self.game_config, state)
        if game.is_terminal():
            raise ValueError("Cannot select an action for a terminal state.")
        legal_actions = game.legal_actions()
        network_calls = 0
        if epsilon == 1 or (epsilon > 0 and self.rng.random() < epsilon):
            action = self.rng.choice(legal_actions)
        else:
            q_values = self.predict_q_values(state)
            action = max(legal_actions, key=lambda candidate: q_values[candidate])
            network_calls = 1
        return AgentDecision(
            action=action,
            stats=SearchStats(
                elapsed_seconds=perf_counter() - started,
                network_calls=network_calls,
            ),
        )

    def predict_q_values(self, state: GameState) -> tuple[float, ...]:
        import torch

        game = Game.from_state(self.game_config, state)
        if game.is_terminal():
            raise ValueError("Cannot predict Q-values for a terminal state.")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                values = self.model(
                    encode_state(state, state.to_play, self.game_config).to(
                        self.device
                    )
                )[0]
            q_values = tuple(float(value) for value in values.cpu().tolist())
        finally:
            self.model.train(was_training)
        if len(q_values) != self.game_config.rows * self.game_config.cols or any(
            not math.isfinite(value) for value in q_values
        ):
            raise FloatingPointError("The network produced invalid Q-values.")
        return q_values

    def _collect_transition(
        self,
        game: Game,
        action: Action,
        learner_player: Player,
        opponent: Agent,
    ) -> Transition:
        if not isinstance(game, Game) or game.config != self.game_config:
            raise ValueError("game must use this agent's GameConfig.")
        if type(learner_player) is not int or learner_player not in (-1, 1):
            raise ValueError("learner_player must be -1 or 1.")
        if not callable(getattr(opponent, "select_action", None)):
            raise TypeError("opponent must implement select_action.")
        if getattr(opponent, "game_config", self.game_config) != self.game_config:
            raise ValueError("Opponent uses a different game configuration.")
        if game.is_terminal() or game.snapshot().to_play != learner_player:
            raise ValueError("The live game must be at the learner's turn.")
        if type(action) is not int or action not in game.legal_actions():
            raise ValueError("Learner action must be legal.")

        state = game.snapshot()
        learner_result = game.step(action)
        reward = 0.0
        if learner_result.outcome.terminal:
            reward = (
                1.0
                if learner_result.outcome.winner == learner_player
                else 0.0
            )
        else:
            opponent_state = game.snapshot()
            opponent_decision = opponent.select_action(
                opponent_state, getattr(opponent, "budget", None)
            )
            if not isinstance(opponent_decision, AgentDecision):
                raise TypeError(
                    "Opponent.select_action must return an AgentDecision."
                )
            if (
                type(opponent_decision.action) is not int
                or opponent_decision.action not in game.legal_actions()
            ):
                raise ValueError("Opponent selected an illegal action.")
            game.step(opponent_decision.action)
            if game.is_terminal():
                reward = (
                    -1.0
                    if game.result().winner == -learner_player
                    else 0.0
                )

        next_state = game.snapshot()
        done = next_state.outcome.terminal
        next_legal_actions = game.legal_actions()
        if not done and next_state.to_play != learner_player:
            raise RuntimeError("Transition did not return to the learner's turn.")
        return Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_legal_actions=next_legal_actions,
            learner_player=learner_player,
        )

    def _learn_batch(self, batch: tuple[Transition, ...]) -> float:
        import torch
        from torch import nn

        if not isinstance(batch, tuple) or not batch:
            raise ValueError("batch must be a non-empty tuple.")
        if self.target_model is None or self.optimizer is None:
            raise RuntimeError("Learning state is unavailable.")

        states = []
        next_states = []
        actions = []
        next_q_values = []
        for transition in batch:
            if not isinstance(transition, Transition):
                raise ValueError("batch entries must be Transition objects.")
            current_game = Game.from_state(self.game_config, transition.state)
            Game.from_state(self.game_config, transition.next_state)
            if transition.state.to_play != transition.learner_player:
                raise ValueError("Transition state must be the learner's turn.")
            if (
                type(transition.action) is not int
                or transition.action not in current_game.legal_actions()
            ):
                raise ValueError("Transition action must be legal in its state.")
            if (
                type(transition.reward) not in (int, float)
                or not math.isfinite(transition.reward)
                or type(transition.done) is not bool
            ):
                raise ValueError("Transition reward or done flag is invalid.")
            states.append(
                encode_state(
                    transition.state,
                    transition.learner_player,
                    self.game_config,
                )
            )
            next_states.append(
                encode_state(
                    transition.next_state,
                    transition.learner_player,
                    self.game_config,
                )
            )
            actions.append(transition.action)

        state_batch = torch.cat(states, dim=0).to(self.device)
        next_state_batch = torch.cat(next_states, dim=0).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        with torch.no_grad():
            target_values = self.target_model(next_state_batch).cpu().tolist()
        next_q_values = tuple(
            tuple(float(value) for value in row) for row in target_values
        )
        target_batch = torch.tensor(
            compute_td_targets(batch, next_q_values, self.config.gamma),
            dtype=torch.float32,
            device=self.device,
        )

        self.model.train()
        current_q = self.model(state_batch).gather(
            1, action_batch.unsqueeze(1)
        ).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current_q, target_batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("DQN loss is not finite.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )
        if not bool(torch.isfinite(gradient_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("DQN gradients are not finite.")
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.optimizer_steps % self.config.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("DQN loss is not finite.")
        return loss_value

    def _play_episode(
        self, opponent: Agent, learner_player: Player
    ) -> dict[str, object]:
        if not callable(getattr(opponent, "select_action", None)):
            raise TypeError("opponent must implement select_action.")
        if getattr(opponent, "game_config", self.game_config) != self.game_config:
            raise ValueError("Opponent uses a different game configuration.")
        if type(learner_player) is not int or learner_player not in (-1, 1):
            raise ValueError("learner_player must be -1 or 1.")

        game = Game(self.game_config)
        if learner_player == -1:
            opening_state = game.snapshot()
            opening_decision = opponent.select_action(
                opening_state, getattr(opponent, "budget", None)
            )
            if not isinstance(opening_decision, AgentDecision):
                raise TypeError(
                    "Opponent.select_action must return an AgentDecision."
                )
            if (
                type(opening_decision.action) is not int
                or opening_decision.action not in game.legal_actions()
            ):
                raise ValueError("Opponent selected an illegal opening action.")
            game.step(opening_decision.action)

        losses = []
        learner_moves = 0
        final_epsilon = self.config.epsilon_start
        while not game.is_terminal():
            state = game.snapshot()
            if state.to_play != learner_player:
                raise RuntimeError("Episode state is not at the learner's turn.")
            progress = min(self.global_step / self.config.epsilon_decay_steps, 1.0)
            final_epsilon = self.config.epsilon_start + progress * (
                self.config.epsilon_end - self.config.epsilon_start
            )
            decision = self.select_action(state, epsilon=final_epsilon)
            transition = self._collect_transition(
                game, decision.action, learner_player, opponent
            )
            if self.replay is None:
                raise RuntimeError("Replay memory is unavailable.")
            self.replay.append(transition)
            self.global_step += 1
            learner_moves += 1

            if len(self.replay) >= max(
                self.config.replay_warmup, self.config.batch_size
            ):
                batch = tuple(
                    self.rng.sample(self.replay, self.config.batch_size)
                )
                losses.append(self._learn_batch(batch))
            if transition.done:
                break

        self.completed_episodes += 1
        result = game.result()
        outcome = (
            "win"
            if result.winner == learner_player
            else "loss"
            if result.winner == -learner_player
            else "draw"
        )
        return {
            "episode": self.completed_episodes,
            "learner_player": learner_player,
            "outcome": outcome,
            "reward": 1.0 if outcome == "win" else -1.0 if outcome == "loss" else 0.0,
            "moves": learner_moves,
            "epsilon": final_epsilon,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
        }

    def train(
        self,
        opponent: Agent,
        validation_opponents: tuple[AgentEntry, ...],
        episodes: int = 1000,
        output_dir: Path = Path("outputs/train"),
        resume_from: Path | None = None,
    ) -> Path:
        from .evaluation import AgentEntry

        if not callable(getattr(opponent, "select_action", None)):
            raise TypeError("opponent must implement select_action.")
        if getattr(opponent, "game_config", self.game_config) != self.game_config:
            raise ValueError("Opponent uses a different game configuration.")
        if (
            not isinstance(validation_opponents, tuple)
            or any(not isinstance(entry, AgentEntry) for entry in validation_opponents)
        ):
            raise ValueError("validation_opponents must be a tuple of AgentEntry.")
        if type(episodes) is not int or episodes < 1:
            raise ValueError("episodes must be a positive integer.")
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a Path.")
        if resume_from is not None and not isinstance(resume_from, Path):
            raise ValueError("resume_from must be a Path or None.")
        for entry in validation_opponents:
            if (
                not isinstance(entry.name, str)
                or not entry.name.strip()
                or not callable(entry.factory)
            ):
                raise ValueError("Validation entries need a name and callable factory.")
            if entry.budget is not None and not isinstance(
                entry.budget, SearchBudget
            ):
                raise ValueError("Validation entry budget must be SearchBudget.")
        training_metadata = {
            "opponent": {
                "class": f"{type(opponent).__module__}.{type(opponent).__qualname__}",
                "name": getattr(opponent, "name", None),
            },
            "validation_opponents": [
                {
                    "name": entry.name,
                    "metadata": entry.metadata,
                    "budget": asdict(entry.budget)
                    if entry.budget is not None
                    else None,
                }
                for entry in validation_opponents
            ],
            "learner_player_schedule": "alternate_x_o",
            "validation_schedule": {
                "every_episodes": self.config.validation_every,
                "games_per_opponent": self.config.validation_games,
                "seeds": self.config.validation_seeds,
            },
        }
        json.dumps(training_metadata, allow_nan=False)
        if resume_from is not None:
            self.load_checkpoint(resume_from, resume=True)
            saved_training = self.metadata.get("training")
            if saved_training is not None and saved_training != training_metadata:
                raise ValueError(
                    "Resume requires the same opponent and validation protocol."
                )
        self.metadata["training"] = training_metadata

        output_dir.mkdir(parents=True, exist_ok=True)
        latest_path = output_dir / "latest.pt"
        metrics_path = output_dir / "training.jsonl"
        for episode_index in range(episodes):
            learner_player: Player = 1 if episode_index % 2 == 0 else -1
            metrics = self._play_episode(opponent, learner_player)

            if self.completed_episodes % self.config.validation_every == 0:
                validation_results = []
                all_scores = []
                for entry in validation_opponents:
                    wins = draws = losses = 0
                    scores = []
                    for game_index in range(self.config.validation_games):
                        seed = self.config.validation_seeds[
                            game_index % len(self.config.validation_seeds)
                        ]
                        validation_opponent = entry.factory(seed)
                        if not callable(
                            getattr(validation_opponent, "select_action", None)
                        ):
                            raise TypeError(
                                "Validation factory must return an agent."
                            )
                        if getattr(
                            validation_opponent, "game_config", self.game_config
                        ) != self.game_config:
                            raise ValueError(
                                "Validation agent uses a different game config."
                            )
                        player: Player = 1 if game_index % 2 == 0 else -1
                        validation_game = Game(self.game_config)
                        if player == -1:
                            opponent_decision = validation_opponent.select_action(
                                validation_game.snapshot(), entry.budget
                            )
                            if not isinstance(opponent_decision, AgentDecision):
                                raise TypeError(
                                    "Validation agent must return AgentDecision."
                                )
                            if (
                                type(opponent_decision.action) is not int
                                or opponent_decision.action
                                not in validation_game.legal_actions()
                            ):
                                raise ValueError(
                                    "Validation agent selected an illegal action."
                                )
                            validation_game.step(opponent_decision.action)
                        while not validation_game.is_terminal():
                            state = validation_game.snapshot()
                            if state.to_play == player:
                                decision = self.select_action(state, epsilon=0.0)
                            else:
                                decision = validation_opponent.select_action(
                                    state, entry.budget
                                )
                            if not isinstance(decision, AgentDecision):
                                raise TypeError(
                                    "Validation agent must return AgentDecision."
                                )
                            if (
                                type(decision.action) is not int
                                or decision.action
                                not in validation_game.legal_actions()
                            ):
                                raise ValueError(
                                    "Validation agent selected an illegal action."
                                )
                            validation_game.step(decision.action)

                        winner = validation_game.result().winner
                        score = (
                            1.0
                            if winner == player
                            else 0.0
                            if winner is None
                            else -1.0
                        )
                        scores.append(score)
                        all_scores.append(score)
                        if score > 0:
                            wins += 1
                        elif score < 0:
                            losses += 1
                        else:
                            draws += 1
                    validation_results.append(
                        {
                            "agent": entry.name,
                            "games": len(scores),
                            "wins": wins,
                            "draws": draws,
                            "losses": losses,
                            "mean_score": sum(scores) / len(scores),
                        }
                    )
                validation_score = (
                    sum(all_scores) / len(all_scores) if all_scores else None
                )
                metrics["validation"] = validation_results
                metrics["validation_score"] = validation_score
                if validation_score is not None:
                    best_score = self.metadata["best_validation_score"]
                    if best_score is None or validation_score > best_score:
                        self.metadata["best_validation_score"] = validation_score
                        self.save_checkpoint(output_dir / "best.pt")

            serialized_metrics = json.dumps(
                metrics,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(serialized_metrics + "\n")
            if self.completed_episodes % self.config.checkpoint_every == 0:
                self.save_checkpoint(latest_path)

        self.save_checkpoint(latest_path)
        return latest_path

    def save_checkpoint(self, path: Path) -> None:
        import torch

        if not isinstance(path, Path):
            raise ValueError("path must be a Path.")
        if self.target_model is None or self.optimizer is None or self.replay is None:
            raise RuntimeError("Checkpoint state is unavailable.")

        replay_data = []
        for transition in self.replay:
            replay_data.append(
                {
                    "state": asdict(transition.state),
                    "action": transition.action,
                    "reward": transition.reward,
                    "next_state": asdict(transition.next_state),
                    "done": transition.done,
                    "next_legal_actions": transition.next_legal_actions,
                    "learner_player": transition.learner_player,
                }
            )
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "game_config": asdict(self.game_config),
            "dqn_config": asdict(self.config),
            "seed": self.seed,
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": replay_data,
            "completed_episodes": self.completed_episodes,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "python_rng_state": self.rng.getstate(),
            "metadata": self.metadata,
        }
        json.dumps(payload["metadata"], allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load_checkpoint(self, path: Path, *, resume: bool = False) -> None:
        import torch

        if not isinstance(path, Path) or not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        if type(resume) is not bool:
            raise ValueError("resume must be a boolean.")
        payload = torch.load(
            path, map_location=self.device, weights_only=True
        )
        if not isinstance(payload, dict) or payload.get(
            "schema_version"
        ) != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported or invalid DQN checkpoint.")
        if payload.get("game_config") != asdict(self.game_config):
            raise ValueError("Checkpoint game configuration does not match.")
        checkpoint_config = payload.get("dqn_config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Checkpoint DQN configuration is invalid.")
        if checkpoint_config.get("encoding_version") != self.config.encoding_version:
            raise ValueError("Checkpoint state encoding does not match.")
        if resume and checkpoint_config != asdict(self.config):
            raise ValueError("Resume requires an identical DQN configuration.")
        if (
            type(payload.get("completed_episodes")) is not int
            or payload["completed_episodes"] < 0
            or type(payload.get("global_step")) is not int
            or payload["global_step"] < 0
            or type(payload.get("optimizer_steps")) is not int
            or payload["optimizer_steps"] < 0
        ):
            raise ValueError("Checkpoint counters are invalid.")
        if type(payload.get("seed")) is not int:
            raise ValueError("Checkpoint seed is invalid.")
        if not isinstance(payload.get("metadata"), dict):
            raise ValueError("Checkpoint metadata is invalid.")

        self.model.load_state_dict(payload["model"])
        if self.target_model is None or self.optimizer is None or self.replay is None:
            raise RuntimeError("Checkpoint target state is unavailable.")
        self.target_model.load_state_dict(payload["target_model"])
        self.target_model.eval()
        self.metadata = payload["metadata"]
        if not resume:
            return

        if not isinstance(payload.get("replay"), list):
            raise ValueError("Checkpoint replay memory is invalid.")
        if len(payload["replay"]) > self.config.replay_capacity:
            raise ValueError("Checkpoint replay exceeds configured capacity.")
        replay = deque(maxlen=self.config.replay_capacity)
        for item in payload["replay"]:
            if not isinstance(item, dict):
                raise ValueError("Checkpoint transition is invalid.")
            state_data = item.get("state")
            next_state_data = item.get("next_state")
            if not isinstance(state_data, dict) or not isinstance(
                next_state_data, dict
            ):
                raise ValueError("Checkpoint transition states are invalid.")

            def restore_state(state_value: dict[str, object]) -> GameState:
                outcome_data = state_value["outcome"]
                if not isinstance(outcome_data, dict):
                    raise ValueError("Checkpoint state outcome is invalid.")
                restored = GameState(
                    board=tuple(
                        tuple(row) for row in state_value["board"]
                    ),
                    to_play=state_value["to_play"],
                    move_count=state_value["move_count"],
                    last_action=state_value["last_action"],
                    outcome=GameResult(**outcome_data),
                )
                Game.from_state(self.game_config, restored)
                return restored

            state = restore_state(state_data)
            next_state = restore_state(next_state_data)
            legal_actions = item.get("next_legal_actions")
            if not isinstance(legal_actions, (tuple, list)):
                raise ValueError("Checkpoint next legal actions are invalid.")
            current_game = Game.from_state(self.game_config, state)
            next_game = Game.from_state(self.game_config, next_state)
            if (
                type(item.get("action")) is not int
                or item["action"] not in current_game.legal_actions()
                or type(item.get("reward")) not in (int, float)
                or not math.isfinite(item["reward"])
                or item["reward"] not in (-1, 0, 1)
                or type(item.get("done")) is not bool
                or item["done"] != next_game.is_terminal()
                or type(item.get("learner_player")) is not int
                or item["learner_player"] not in (-1, 1)
                or state.to_play != item["learner_player"]
                or tuple(legal_actions) != next_game.legal_actions()
                or (
                    not item["done"]
                    and next_state.to_play != item["learner_player"]
                )
            ):
                raise ValueError("Checkpoint transition is inconsistent.")
            transition = Transition(
                state=state,
                action=item.get("action"),
                reward=item.get("reward"),
                next_state=next_state,
                done=item.get("done"),
                next_legal_actions=tuple(legal_actions),
                learner_player=item.get("learner_player"),
            )
            replay.append(transition)

        self.optimizer.load_state_dict(payload["optimizer"])
        self.replay = replay
        self.completed_episodes = payload["completed_episodes"]
        self.global_step = payload["global_step"]
        self.optimizer_steps = payload["optimizer_steps"]
        self.seed = payload["seed"]
        try:
            self.rng.setstate(payload["python_rng_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("Checkpoint random generator state is invalid.") from error
