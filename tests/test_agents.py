import math
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ai.agents import (
    MinimaxAgent,
    RandomAgent,
    RuleAgent,
    evaluate_position,
    winning_actions,
)
from ttt_ai.game import Game, GameConfig, SearchBudget


def game_after_moves(moves, config=GameConfig()):
    game = Game(config)
    for action in moves:
        game.step(action)
    return game


class AgentTests(unittest.TestCase):
    def test_random_agent_is_seeded_legal_and_does_not_use_global_rng(self):
        state = Game().snapshot()
        random.seed(1234)
        global_state = random.getstate()
        first = RandomAgent(GameConfig(), seed=42).select_action(state)
        second = RandomAgent(GameConfig(), seed=42).select_action(state)
        self.assertIn(first.action, range(42))
        self.assertEqual(first.action, second.action)
        self.assertIsNone(first.estimated_value)
        self.assertEqual(random.getstate(), global_state)

    def test_random_agent_rejects_terminal_state_and_budget(self):
        agent = RandomAgent(GameConfig())
        terminal = game_after_moves((0, 7, 1, 8, 2, 9, 3)).snapshot()
        with self.assertRaises(ValueError):
            agent.select_action(terminal)
        with self.assertRaises(ValueError):
            agent.select_action(Game().snapshot(), SearchBudget(max_depth=1))

    def test_winning_actions_finds_immediate_wins_for_both_players(self):
        game = game_after_moves((0, 7, 1, 8, 2, 9))
        state = game.snapshot()
        self.assertEqual(winning_actions(state, game.config, 1), (3,))
        self.assertEqual(winning_actions(state, game.config, -1), (10,))
        self.assertIs(game.snapshot(), state)

    def test_winning_actions_returns_empty_for_terminal_state(self):
        game = game_after_moves((0, 7, 1, 8, 2, 9, 3))
        self.assertEqual(winning_actions(game.snapshot(), game.config, -1), ())

    def test_evaluate_position_is_bounded_and_changes_sign_with_perspective(self):
        game = game_after_moves((0, 8, 1, 9))
        state = game.snapshot()
        x_value = evaluate_position(state, game.config, 1)
        o_value = evaluate_position(state, game.config, -1)
        self.assertTrue(math.isfinite(x_value))
        self.assertLessEqual(abs(x_value), 0.99)
        self.assertAlmostEqual(x_value, -o_value)
        self.assertEqual(evaluate_position(Game().snapshot(), GameConfig(), 1), 0.0)

    def test_evaluate_position_rejects_terminal_state(self):
        game = game_after_moves((0, 7, 1, 8, 2, 9, 3))
        with self.assertRaises(ValueError):
            evaluate_position(game.snapshot(), game.config, 1)

    def test_rule_agent_wins_before_blocking(self):
        game = game_after_moves((0, 7, 1, 8, 2, 9))
        decision = RuleAgent(game.config).select_action(game.snapshot())
        self.assertEqual(decision.action, 3)
        self.assertEqual(decision.estimated_value, 1.0)

    def test_rule_agent_blocks_an_immediate_loss(self):
        game = game_after_moves((7, 0, 9, 1, 14, 2))
        decision = RuleAgent(game.config).select_action(game.snapshot())
        self.assertEqual(decision.action, 3)

    def test_rule_agent_uses_heuristic_and_lowest_action_for_ties(self):
        decision = RuleAgent(GameConfig()).select_action(Game().snapshot())
        self.assertEqual(decision.action, 17)
        self.assertIsNotNone(decision.estimated_value)

    def test_rule_agent_rejects_search_budget(self):
        with self.assertRaises(ValueError):
            RuleAgent(GameConfig()).select_action(
                Game().snapshot(), SearchBudget(time_limit_seconds=1.0)
            )

    def test_minimax_takes_immediate_win(self):
        game = game_after_moves((0, 7, 1, 8, 2, 9))
        decision = MinimaxAgent(game.config, depth=1).select_action(game.snapshot())
        self.assertEqual(decision.action, 3)
        self.assertEqual(decision.estimated_value, 1.0)
        self.assertEqual(decision.stats.completed_depth, 1)

    def test_minimax_blocks_immediate_loss_at_depth_two(self):
        game = game_after_moves((7, 0, 9, 1, 14, 2))
        decision = MinimaxAgent(game.config, depth=2).select_action(game.snapshot())
        self.assertEqual(decision.action, 3)

    def test_minimax_node_counts_match_full_tree_without_alpha_beta(self):
        game = Game()
        depth_one = MinimaxAgent(game.config, depth=1, use_alpha_beta=False)
        depth_two = MinimaxAgent(game.config, depth=2, use_alpha_beta=False)
        self.assertEqual(
            depth_one.select_action(game.snapshot()).stats.nodes_visited, 43
        )
        self.assertEqual(
            depth_two.select_action(game.snapshot()).stats.nodes_visited, 1765
        )

    def test_alpha_beta_preserves_root_decision_and_reduces_or_matches_nodes(self):
        game = Game()
        plain = MinimaxAgent(game.config, depth=2, use_alpha_beta=False)
        pruned = MinimaxAgent(game.config, depth=2, use_alpha_beta=True)
        plain_decision = plain.select_action(game.snapshot())
        pruned_decision = pruned.select_action(game.snapshot())
        self.assertEqual(pruned_decision.action, plain_decision.action)
        self.assertEqual(
            pruned_decision.estimated_value, plain_decision.estimated_value
        )
        self.assertLessEqual(
            pruned_decision.stats.nodes_visited,
            plain_decision.stats.nodes_visited,
        )

    def test_minimax_terminal_values_are_checked_before_depth_cutoff(self):
        won_game = game_after_moves((0, 7, 1, 8, 2, 9, 3))
        agent = MinimaxAgent(won_game.config, depth=1)
        self.assertEqual(agent._search(won_game, 0, 1, -math.inf, math.inf), 1.0)
        self.assertEqual(agent._search(won_game, 0, -1, -math.inf, math.inf), -1.0)

        draw_game = game_after_moves((2, 0, 3, 1, 4, 5), GameConfig(2, 3, 3))
        draw_agent = MinimaxAgent(draw_game.config, depth=1)
        self.assertEqual(draw_agent._search(draw_game, 0, 1, -math.inf, math.inf), 0.0)

    def test_minimax_budget_overrides_depth_and_rejects_invalid_values(self):
        game = Game()
        agent = MinimaxAgent(game.config, depth=1, use_alpha_beta=False)
        decision = agent.select_action(
            game.snapshot(), SearchBudget(max_depth=2)
        )
        self.assertEqual(decision.stats.completed_depth, 2)
        self.assertEqual(decision.stats.nodes_visited, 1765)
        for budget in (
            SearchBudget(max_depth=True),
            SearchBudget(max_depth=0),
            SearchBudget(max_simulations=1),
            SearchBudget(time_limit_seconds=0),
            SearchBudget(time_limit_seconds=True),
            SearchBudget(time_limit_seconds=float("inf")),
            SearchBudget(time_limit_seconds=10**10000),
        ):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    agent.select_action(game.snapshot(), budget)

    def test_minimax_timeout_before_depth_one_returns_first_legal_action(self):
        game = Game()
        agent = MinimaxAgent(game.config, depth=2)
        clock_values = iter((0.0, 0.0, 0.0, 1.0))

        def fake_clock():
            return next(clock_values, 1.0)

        with patch("ttt_ai.agents.perf_counter", side_effect=fake_clock):
            decision = agent.select_action(
                game.snapshot(), SearchBudget(time_limit_seconds=0.5)
            )
        self.assertEqual(decision.action, 0)
        self.assertIsNone(decision.estimated_value)
        self.assertEqual(decision.stats.completed_depth, 0)

    def test_minimax_timeout_keeps_last_completed_iteration(self):
        game = Game()
        agent = MinimaxAgent(game.config, depth=2)
        calls = 0

        def search_with_timeout(*args):
            nonlocal calls
            calls += 1
            if calls == len(game.legal_actions()) + 1:
                raise TimeoutError
            return 0.25

        with (
            patch("ttt_ai.agents.perf_counter", return_value=0.0),
            patch.object(agent, "_search", side_effect=search_with_timeout),
        ):
            decision = agent.select_action(
                game.snapshot(), SearchBudget(time_limit_seconds=10.0)
            )
        self.assertEqual(decision.action, 0)
        self.assertEqual(decision.estimated_value, 0.25)
        self.assertEqual(decision.stats.completed_depth, 1)

    def test_agent_constructors_reject_boolean_integer_arguments(self):
        config = GameConfig()
        for constructor in (
            lambda: RandomAgent(config, seed=True),
            lambda: RuleAgent(config, seed=True),
            lambda: MinimaxAgent(config, depth=True),
            lambda: MinimaxAgent(config, use_alpha_beta=1),
            lambda: MinimaxAgent(config, seed=False),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()


if __name__ == "__main__":
    unittest.main()
