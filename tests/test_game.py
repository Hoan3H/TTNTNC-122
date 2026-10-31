from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from io import StringIO
from pathlib import Path
from random import Random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ai.game import Game, GameConfig, GameResult, GameState


class GameTests(unittest.TestCase):
    def test_default_board_and_legal_actions(self):
        game = Game()
        state = game.snapshot()
        self.assertEqual(game.config, GameConfig(6, 7, 4))
        self.assertEqual(state.board, ((0,) * 7,) * 6)
        self.assertEqual((state.to_play, state.move_count, state.last_action), (1, 0, None))
        self.assertEqual(game.result(), GameResult(False, None, False))
        self.assertFalse(game.is_terminal())
        self.assertEqual(game.legal_actions(), tuple(range(42)))

    def test_invalid_configurations(self):
        for field in ("rows", "cols", "win_length"):
            for value in (True, False, 0, -1, 1.5, "3", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        Game(replace(GameConfig(), **{field: value}))
        for config in (False, 0, {}, (), "6x7", GameConfig(2, 3, 4)):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    Game(config)
        self.assertEqual(Game(GameConfig(1, 7, 7)).config.win_length, 7)

    def test_reset_for_either_starting_player(self):
        game = Game()
        previous = game.step(17)
        for player in (-1, 1):
            with self.subTest(player=player):
                state = game.reset(player)
                self.assertEqual(state.to_play, player)
                self.assertEqual(state.move_count, 0)
                self.assertIsNone(state.last_action)
                self.assertFalse(state.outcome.terminal)
                self.assertEqual(len(game.legal_actions()), 42)
        self.assertEqual(previous.board[2][3], 1)

    def test_invalid_reset_preserves_state(self):
        game = Game()
        previous = game.step(17)
        for player in (True, False, 0, 2, -2, 1.0, "1", None):
            with self.subTest(player=player):
                with self.assertRaises(ValueError):
                    game.reset(player)
                self.assertIs(game.snapshot(), previous)

    def test_action_mapping_and_turn_changes(self):
        game = Game()
        before = game.snapshot()
        after = game.step(17)
        self.assertEqual(after.board[2][3], 1)
        self.assertEqual((after.to_play, after.move_count, after.last_action), (-1, 1, 17))
        self.assertEqual(len(game.legal_actions()), 41)
        self.assertNotIn(17, game.legal_actions())
        self.assertEqual(before.board[2][3], 0)
        self.assertEqual(game.step(0).board[0][0], -1)
        self.assertEqual(game.snapshot().to_play, 1)

    def test_snapshot_is_immutable(self):
        state = Game().snapshot()
        with self.assertRaises(FrozenInstanceError):
            state.to_play = -1
        with self.assertRaises(TypeError):
            state.board[0][0] = 1
        with self.assertRaises(FrozenInstanceError):
            state.outcome.terminal = True

    def test_invalid_actions_preserve_state(self):
        game = Game()
        previous = game.step(17)
        for action in (-1, 42, 100, True, False, 1.5, "1", None, 17):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    game.step(action)
                self.assertIs(game.snapshot(), previous)

    def test_failed_win_check_does_not_commit_candidate_board(self):
        game = Game()
        previous = game.snapshot()
        with patch.object(game, "_check_win", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                game.step(17)
        self.assertIs(game.snapshot(), previous)
        self.assertEqual(len(game.legal_actions()), 42)

    def test_four_win_directions_and_terminal_behavior(self):
        sequences = (
            (0, 7, 1, 8, 2, 9, 3),
            (0, 1, 7, 2, 14, 3, 21),
            (0, 1, 8, 2, 16, 3, 24),
            (3, 0, 9, 1, 15, 2, 21),
        )
        for starting_player in (1, -1):
            for sequence in sequences:
                with self.subTest(player=starting_player, sequence=sequence):
                    game = Game()
                    game.reset(starting_player)
                    for action in sequence[:-1]:
                        game.step(action)
                        self.assertFalse(game.is_terminal())
                    state = game.step(sequence[-1])
                    self.assertEqual(state.outcome, GameResult(True, starting_player, False))
                    self.assertEqual(state.to_play, -starting_player)
                    self.assertEqual(game.legal_actions(), ())
                    self.assertTrue(game._check_win(sequence[-1], starting_player))
                    with self.assertRaises(ValueError):
                        game.step(41)
                    self.assertIs(game.snapshot(), state)

    def test_second_player_can_win(self):
        game = Game()
        for action in (7, 0, 9, 1, 11, 2, 13, 3):
            game.step(action)
        self.assertEqual(game.result(), GameResult(True, -1, False))
        self.assertEqual(game.snapshot().to_play, 1)

    def test_filling_gap_can_create_more_than_k(self):
        game = Game()
        for action in (0, 7, 1, 9, 3, 11, 4, 13):
            game.step(action)
            self.assertFalse(game.is_terminal())
        state = game.step(2)
        self.assertEqual(state.board[0][:5], (1, 1, 1, 1, 1))
        self.assertEqual(state.outcome, GameResult(True, 1, False))
        self.assertEqual(Game.from_state(game.config, state).snapshot(), state)

    def test_row_boundary_is_not_a_line(self):
        game = Game(GameConfig(2, 3, 3))
        for action in (2, 0, 3, 1, 4):
            game.step(action)
        self.assertFalse(game.is_terminal())
        self.assertEqual(game.legal_actions(), (5,))
        self.assertEqual(game.step(5).outcome, GameResult(True, None, True))

    def test_draw_and_winning_last_cell(self):
        cases = (
            ((0, 1, 2, 4, 3, 5, 7, 6, 8), GameResult(True, None, True)),
            ((0, 4, 1, 5, 3, 6, 7, 8, 2), GameResult(True, 1, False)),
        )
        for sequence, expected in cases:
            with self.subTest(sequence=sequence):
                game = Game(GameConfig(3, 3, 3))
                for action in sequence:
                    game.step(action)
                self.assertEqual(game.result(), expected)
                self.assertEqual(game.snapshot().move_count, 9)
                self.assertEqual(game.legal_actions(), ())
                self.assertEqual(Game.from_state(game.config, game.snapshot()).result(), expected)

    def test_k_one_ends_after_first_move(self):
        for config in (GameConfig(1, 1, 1), GameConfig(2, 3, 1)):
            with self.subTest(config=config):
                game = Game(config)
                state = game.step(0)
                self.assertEqual(state.outcome, GameResult(True, 1, False))
                self.assertEqual(game.legal_actions(), ())
                self.assertEqual(Game.from_state(config, state).snapshot(), state)

    def test_candidate_board_check_does_not_modify_game(self):
        game = Game(GameConfig(2, 3, 3))
        state = game.snapshot()
        board = ((1, 1, 1), (0, 0, 0))
        self.assertTrue(game._check_win(1, 1, board=board))
        self.assertIs(game.snapshot(), state)
        for action, player, candidate in (
            (-1, 1, board), (6, 1, board), (True, 1, board),
            (0, True, board), (0, 0, board), (0, -1, board),
            (0, 1, ((1, 1),)), (0, 1, [[1, 1, 1], [0, 0, 0]]),
        ):
            with self.subTest(action=action, player=player, board=candidate):
                with self.assertRaises(ValueError):
                    game._check_win(action, player, board=candidate)

    def test_clone_moves_and_resets_independently(self):
        game = Game()
        state = game.step(17)
        clone = game.clone()
        self.assertIsNot(clone, game)
        self.assertEqual(clone.config, game.config)
        self.assertEqual(clone.snapshot(), state)
        clone.step(0)
        self.assertIs(game.snapshot(), state)
        clone.reset(-1)
        self.assertIs(game.snapshot(), state)
        self.assertEqual(clone.snapshot().move_count, 0)

    def test_round_trip_for_both_starting_players(self):
        for player in (1, -1):
            with self.subTest(player=player):
                game = Game()
                game.reset(player)
                states = [game.snapshot()]
                for action in (0, 7, 1, 8, 2, 9, 3):
                    states.append(game.step(action))
                for state in states:
                    loaded = Game.from_state(game.config, state)
                    self.assertEqual(loaded.snapshot(), state)
                    self.assertIsNot(loaded, game)

    def test_from_state_rejects_invalid_fields(self):
        game = Game(GameConfig(3, 3, 3))
        empty = game.snapshot()
        one_move = game.step(0)
        invalid_states = [
            None,
            replace(empty, board=[]),
            replace(empty, board=((0, 0, 0),)),
            replace(empty, board=([0, 0, 0], (0, 0, 0), (0, 0, 0))),
            replace(empty, board=((0, 0), (0, 0), (0, 0))),
            replace(empty, board=((True, 0, 0), (0, 0, 0), (0, 0, 0))),
            replace(empty, board=((2, 0, 0), (0, 0, 0), (0, 0, 0))),
            replace(empty, board=((0.0, 0, 0), (0, 0, 0), (0, 0, 0))),
            replace(empty, to_play=True),
            replace(empty, to_play=0),
            replace(empty, move_count=True),
            replace(empty, move_count=-1),
            replace(empty, move_count=1),
            replace(empty, last_action=0),
            replace(empty, outcome=None),
            replace(empty, outcome=GameResult(0, None, False)),
            replace(empty, outcome=GameResult(False, None, 0)),
            replace(empty, outcome=GameResult(True, None, False)),
            replace(empty, outcome=GameResult(True, None, True)),
            replace(empty, outcome=GameResult(True, True, False)),
            replace(empty, outcome=GameResult(True, 2, False)),
            replace(one_move, to_play=1),
            replace(one_move, last_action=None),
            replace(one_move, last_action=True),
            replace(one_move, last_action=9),
            replace(one_move, last_action=1),
            replace(one_move, outcome=GameResult(True, 1, False)),
        ]
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    Game.from_state(game.config, state)
        for config in (None, {}, GameConfig(0, 3, 3)):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    Game.from_state(config, empty)
        self.assertIs(game.snapshot(), one_move)

    def test_from_state_rejects_two_winners(self):
        state = GameState(
            ((1, 1, 1), (-1, -1, -1), (0, 0, 0)),
            1, 6, 5, GameResult(True, -1, False),
        )
        with self.assertRaises(ValueError):
            Game.from_state(GameConfig(3, 3, 3), state)

    def test_from_state_rejects_moves_after_an_earlier_win(self):
        state = GameState(
            ((1, 1, 1, 0), (-1, 0, 1, -1), (0, 0, 1, 0), (-1, 0, 0, -1)),
            -1, 9, 10, GameResult(True, 1, False),
        )
        with self.assertRaises(ValueError):
            Game.from_state(GameConfig(4, 4, 3), state)

    def test_from_state_requires_win_through_last_action(self):
        state = GameState(
            ((1, 1, 1), (-1, -1, 0), (1, -1, 0)),
            -1, 7, 6, GameResult(True, 1, False),
        )
        with self.assertRaises(ValueError):
            Game.from_state(GameConfig(3, 3, 3), state)

    def test_simultaneous_lines_created_by_one_move_are_valid(self):
        game = Game(GameConfig(3, 3, 3))
        for action in (1, 0, 3, 2, 5, 6, 7, 8, 4):
            game.step(action)
        self.assertEqual(game.result(), GameResult(True, 1, False))
        self.assertEqual(Game.from_state(game.config, game.snapshot()).snapshot(), game.snapshot())

    def test_inconsistent_outcomes_and_winning_turn_are_rejected(self):
        game = Game(GameConfig(3, 3, 3))
        for action in (0, 3, 1, 4, 2):
            game.step(action)
        won = game.snapshot()
        for state in (
            replace(won, outcome=GameResult(False, None, False)),
            replace(won, outcome=GameResult(True, None, True)),
            replace(won, outcome=GameResult(True, -1, False)),
            replace(won, outcome=GameResult(True, 1, True)),
            replace(won, to_play=1),
        ):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    Game.from_state(game.config, state)

    def test_render_uses_one_based_coordinates_and_does_not_print(self):
        game = Game(GameConfig(2, 3, 3))
        state = game.step(5)
        stdout = StringIO()
        with redirect_stdout(stdout):
            rendered = game.render()
        self.assertEqual(rendered, "  1 2 3\n1 . . .\n2 . . X\n\nTurn: O")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIs(game.snapshot(), state)
        large_board = Game(GameConfig(2, 12, 4)).render()
        self.assertIn("12", large_board.splitlines()[0])
        for sequence, status in (
            ((0, 3, 1, 4, 2), "Winner: X"),
            ((0, 1, 2, 4, 3, 5, 7, 6, 8), "Draw"),
        ):
            game = Game(GameConfig(3, 3, 3))
            for action in sequence:
                game.step(action)
            self.assertTrue(game.render().endswith(status))

    def test_random_legal_games_round_trip_every_state(self):
        rng = Random(20261031)
        for config in (GameConfig(2, 3, 3), GameConfig(3, 3, 3), GameConfig(6, 7, 4)):
            for player in (1, -1):
                for episode in range(10):
                    with self.subTest(config=config, player=player, episode=episode):
                        game = Game(config)
                        game.reset(player)
                        while not game.is_terminal():
                            before = game.snapshot()
                            actions = game.legal_actions()
                            self.assertEqual(len(actions), config.rows * config.cols - before.move_count)
                            state = game.step(rng.choice(actions))
                            self.assertEqual(state.move_count, before.move_count + 1)
                            self.assertEqual(state.to_play, -before.to_play)
                            loaded = Game.from_state(config, state)
                            self.assertEqual(loaded.snapshot(), state)
                        self.assertEqual(game.legal_actions(), ())


if __name__ == "__main__":
    unittest.main()
