import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ai.agents import RandomAgent
from ttt_ai.dqn import (
    DQNAgent,
    DQNConfig,
    Transition,
    build_network,
    compute_td_targets,
    encode_state,
)
from ttt_ai.evaluation import AgentEntry
from ttt_ai.game import AgentDecision, Game, GameConfig, SearchBudget, SearchStats


class FixedActionAgent:
    def __init__(self, game_config, actions):
        self.game_config = game_config
        self.actions = iter(actions)

    def select_action(self, state, budget=None):
        return AgentDecision(
            action=next(self.actions),
            stats=SearchStats(elapsed_seconds=0.0),
        )


def game_after_moves(config, moves):
    game = Game(config)
    for action in moves:
        game.step(action)
    return game


def small_dqn_config(**overrides):
    values = {
        "batch_size": 2,
        "replay_capacity": 16,
        "replay_warmup": 2,
        "target_update_steps": 1,
        "epsilon_decay_steps": 8,
        "checkpoint_every": 1,
        "validation_every": 1,
        "validation_games": 2,
        "validation_seeds": (101, 102),
    }
    values.update(overrides)
    return DQNConfig(**values)


class DQNTests(unittest.TestCase):
    def test_encode_state_uses_two_perspective_channels(self):
        config = GameConfig(2, 3, 3)
        game = game_after_moves(config, (0, 4))
        x_tensor = encode_state(game.snapshot(), 1, config)
        o_tensor = encode_state(game.snapshot(), -1, config)
        self.assertEqual(tuple(x_tensor.shape), (1, 2, 2, 3))
        self.assertEqual(x_tensor.dtype, torch.float32)
        self.assertEqual(float(x_tensor[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(x_tensor[0, 1, 1, 1]), 1.0)
        self.assertTrue(torch.equal(x_tensor[:, 0], o_tensor[:, 1]))
        self.assertTrue(torch.equal(x_tensor[:, 1], o_tensor[:, 0]))

    def test_encode_state_rejects_bad_perspective_or_state(self):
        config = GameConfig(2, 3, 3)
        with self.assertRaises(ValueError):
            encode_state(Game(config).snapshot(), True, config)
        with self.assertRaises(ValueError):
            encode_state(Game(config).snapshot(), 1, GameConfig())

    def test_build_network_supports_variable_board_shapes(self):
        config = GameConfig(2, 3, 3)
        model = build_network(config)
        result = model(encode_state(Game(config).snapshot(), 1, config))
        self.assertEqual(tuple(result.shape), (1, 6))
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_compute_td_targets_masks_illegal_actions_and_handles_terminal(self):
        config = GameConfig(2, 3, 3)
        game = Game(config)
        next_game = game_after_moves(config, (0, 3))
        legal_transition = Transition(
            state=game.snapshot(),
            action=0,
            reward=0.0,
            next_state=next_game.snapshot(),
            done=False,
            next_legal_actions=next_game.legal_actions(),
            learner_player=1,
        )
        terminal_game = game_after_moves(config, (0, 3, 1, 4, 2))
        terminal_transition = Transition(
            state=game.snapshot(),
            action=2,
            reward=1.0,
            next_state=terminal_game.snapshot(),
            done=True,
            next_legal_actions=(),
            learner_player=1,
        )
        q_values = (
            (100.0, 2.0, 99.0, 1.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            compute_td_targets(
                (legal_transition, terminal_transition), q_values, 0.5
            ),
            (49.5, 1.0),
        )

    def test_compute_td_targets_rejects_empty_or_invalid_legal_actions(self):
        config = GameConfig(2, 3, 3)
        state = Game(config).snapshot()
        transition = Transition(state, 0, 0.0, state, False, (), 1)
        with self.assertRaises(ValueError):
            compute_td_targets((transition,), ((0.0,) * 6,), 0.9)
        with self.assertRaises(ValueError):
            compute_td_targets((transition,), ((0.0,) * 6,), True)

    def test_agent_prediction_and_epsilon_greedy_only_select_legal_moves(self):
        config = GameConfig(2, 3, 3)
        game = game_after_moves(config, (0,))
        agent = DQNAgent(config, small_dqn_config(), seed=5)
        with torch.no_grad():
            final_layer = agent.model[-1]
            final_layer.weight.zero_()
            final_layer.bias.copy_(torch.tensor((100.0, 1.0, 2.0, 8.0, 4.0, 3.0)))
        q_values = agent.predict_q_values(game.snapshot())
        self.assertEqual(len(q_values), 6)
        self.assertEqual(max(game.legal_actions(), key=q_values.__getitem__), 3)
        decision = agent.select_action(game.snapshot())
        self.assertEqual(decision.action, 3)
        random_decision = agent.select_action(game.snapshot(), epsilon=1.0)
        self.assertIn(random_decision.action, game.legal_actions())
        self.assertEqual(random_decision.stats.network_calls, 0)
        self.assertEqual(decision.stats.network_calls, 1)

    def test_inference_does_not_consume_agent_rng_or_global_torch_rng(self):
        config = GameConfig(2, 3, 3)
        game = Game(config)
        torch.manual_seed(123)
        global_rng_state = torch.random.get_rng_state().clone()
        agent = DQNAgent(config, small_dqn_config(), seed=7)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_rng_state))
        agent_rng_state = agent.rng.getstate()
        agent.select_action(game.snapshot(), epsilon=0.0)
        self.assertEqual(agent.rng.getstate(), agent_rng_state)

    def test_agent_rejects_terminal_input_and_bad_budget_or_epsilon(self):
        config = GameConfig(2, 3, 3)
        agent = DQNAgent(config, small_dqn_config())
        terminal = game_after_moves(config, (0, 3, 1, 4, 2)).snapshot()
        with self.assertRaises(ValueError):
            agent.select_action(terminal)
        with self.assertRaises(ValueError):
            agent.predict_q_values(terminal)
        with self.assertRaises(ValueError):
            agent.select_action(Game(config).snapshot(), SearchBudget(max_depth=1))
        with self.assertRaises(ValueError):
            agent.select_action(Game(config).snapshot(), epsilon=True)

    def test_collect_transition_rewards_learner_win_and_returns_same_turn_state(self):
        config = GameConfig(3, 3, 3)
        agent = DQNAgent(config, small_dqn_config())
        winning_game = game_after_moves(config, (0, 3, 1, 4))
        win = agent._collect_transition(
            winning_game,
            2,
            1,
            FixedActionAgent(config, (5,)),
        )
        self.assertEqual(win.reward, 1.0)
        self.assertTrue(win.done)
        self.assertEqual(win.next_legal_actions, ())

        ongoing_game = game_after_moves(config, (0, 3))
        ongoing = agent._collect_transition(
            ongoing_game,
            1,
            1,
            FixedActionAgent(config, (4,)),
        )
        self.assertEqual(ongoing.reward, 0.0)
        self.assertFalse(ongoing.done)
        self.assertEqual(ongoing.next_state.to_play, 1)
        self.assertIn(8, ongoing.next_legal_actions)

    def test_collect_transition_assigns_loss_when_opponent_wins(self):
        config = GameConfig(3, 3, 3)
        agent = DQNAgent(config, small_dqn_config())
        game = game_after_moves(config, (0, 3, 1, 4))
        transition = agent._collect_transition(
            game,
            8,
            1,
            FixedActionAgent(config, (5,)),
        )
        self.assertEqual(transition.reward, -1.0)
        self.assertTrue(transition.done)
        self.assertEqual(transition.next_state.outcome.winner, -1)

    def test_learn_batch_returns_finite_loss_and_updates_parameters(self):
        config = GameConfig(2, 3, 3)
        agent = DQNAgent(config, small_dqn_config())
        game = Game(config)
        transition = agent._collect_transition(
            game, 0, 1, FixedActionAgent(config, (3,))
        )
        before = tuple(parameter.detach().clone() for parameter in agent.model.parameters())
        loss = agent._learn_batch((transition, transition))
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 0)
        self.assertEqual(agent.optimizer_steps, 1)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, agent.model.parameters())
            )
        )
        for target, online in zip(
            agent.target_model.parameters(), agent.model.parameters()
        ):
            self.assertTrue(torch.equal(target, online))

    def test_invalid_dqn_configuration_is_rejected(self):
        config = GameConfig(2, 3, 3)
        for dqn_config in (
            DQNConfig(batch_size=True),
            DQNConfig(batch_size=3, replay_capacity=2),
            DQNConfig(replay_warmup=10001),
            DQNConfig(gamma=float("nan")),
            DQNConfig(learning_rate=0),
            DQNConfig(epsilon_start=0.1, epsilon_end=0.9),
            DQNConfig(validation_seeds=()),
        ):
            with self.subTest(dqn_config=dqn_config):
                with self.assertRaises(ValueError):
                    DQNAgent(config, dqn_config)

    def test_checkpoint_round_trip_inference_and_resume_state(self):
        config = GameConfig(2, 3, 3)
        dqn_config = small_dqn_config()
        agent = DQNAgent(config, dqn_config, seed=19)
        game = Game(config)
        transition = agent._collect_transition(
            game, 0, 1, FixedActionAgent(config, (3,))
        )
        agent.replay.append(transition)
        agent.completed_episodes = 2
        agent.global_step = 4
        agent.optimizer_steps = 1
        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            agent.save_checkpoint(checkpoint)
            expected_q_values = agent.predict_q_values(Game(config).snapshot())

            inference_agent = DQNAgent(config, dqn_config, seed=20)
            inference_agent.load_checkpoint(checkpoint)
            self.assertEqual(
                inference_agent.predict_q_values(Game(config).snapshot()),
                expected_q_values,
            )
            self.assertEqual(inference_agent.completed_episodes, 0)

            resumed_agent = DQNAgent(config, dqn_config, seed=20)
            resumed_agent.load_checkpoint(checkpoint, resume=True)
            self.assertEqual(resumed_agent.completed_episodes, 2)
            self.assertEqual(resumed_agent.global_step, 4)
            self.assertEqual(resumed_agent.optimizer_steps, 1)
            self.assertEqual(tuple(resumed_agent.replay), (transition,))

    def test_train_runs_smoke_validation_and_writes_checkpoint_and_metrics(self):
        config = GameConfig(2, 3, 3)
        dqn_config = small_dqn_config(
            batch_size=1,
            replay_warmup=1,
            validation_games=1,
            validation_seeds=(777,),
        )
        agent = DQNAgent(config, dqn_config, seed=31)
        opponent_entry = AgentEntry(
            name="random",
            factory=lambda seed: RandomAgent(config, seed),
            metadata={"kind": "random"},
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "train"
            checkpoint = agent.train(
                RandomAgent(config, seed=99),
                (opponent_entry,),
                episodes=1,
                output_dir=output_dir,
            )
            self.assertEqual(checkpoint, output_dir / "latest.pt")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            metrics = (output_dir / "training.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(metrics), 1)
            self.assertEqual(agent.completed_episodes, 1)
            self.assertGreater(agent.optimizer_steps, 0)
            self.assertIn('"validation_score"', metrics[0])

    def test_training_resumes_checkpoint_and_adds_requested_episodes(self):
        config = GameConfig(2, 3, 3)
        dqn_config = small_dqn_config(
            batch_size=1,
            replay_warmup=1,
            validation_every=2,
            checkpoint_every=1,
        )
        agent = DQNAgent(config, dqn_config, seed=37)
        opponent = RandomAgent(config, seed=17)
        validation_entry = AgentEntry(
            name="random",
            factory=lambda seed: RandomAgent(config, seed),
            metadata={},
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "training"
            initial_checkpoint = agent.train(
                opponent,
                (validation_entry,),
                episodes=1,
                output_dir=output_dir,
            )
            resumed = DQNAgent(config, dqn_config, seed=999)
            final_checkpoint = resumed.train(
                RandomAgent(config, seed=18),
                (validation_entry,),
                episodes=1,
                output_dir=output_dir,
                resume_from=initial_checkpoint,
            )
            self.assertEqual(final_checkpoint, output_dir / "latest.pt")
            self.assertEqual(resumed.completed_episodes, 2)
            self.assertTrue(final_checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
