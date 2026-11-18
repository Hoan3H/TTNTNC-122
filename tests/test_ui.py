from datetime import date
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ai.evaluation import MoveLog, load_logs, save_logs
from ttt_ai.game import Game, GameConfig, SearchBudget, SearchStats
from ttt_ai.ui import _build_match_log


class UITests(unittest.TestCase):
    def test_human_agent_match_log_replays_and_keeps_each_move(self):
        config = GameConfig(rows=2, cols=3, win_length=3)
        game = Game(config)
        actions = (0, 3, 1, 4, 2)
        moves = []
        for action in actions:
            state = game.snapshot()
            game.step(action)
            moves.append(
                MoveLog(
                    state=state,
                    action=action,
                    stats=SearchStats(
                        elapsed_seconds=0.01 if state.to_play == -1 else 0.0
                    ),
                )
            )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ui_games"
            match = _build_match_log(
                game_config=config,
                human_player=1,
                agent_name="Rule",
                agent_seed=27,
                agent_metadata={"kind": "rule"},
                agent_budget=None,
                moves=tuple(moves),
                result=game.result(),
                output_dir=output_dir,
                match_id="ui-match-1",
            )
            save_logs(
                (match,),
                output_dir,
                day=date(2026, 11, 17),
            )
            loaded = load_logs(output_dir, day=date(2026, 11, 17))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(tuple(move.action for move in loaded[0].moves), actions)
        self.assertEqual(loaded[0].agent_x, "Human")
        self.assertEqual(loaded[0].agent_o, "Rule")
        self.assertEqual(loaded[0].protocol.split, "interactive")
        self.assertEqual(loaded[0].seed_o, 27)
        self.assertEqual(loaded[0].agent_metadata["X"]["agent"], {"kind": "human"})
        self.assertEqual(loaded[0].result.winner, 1)

    def test_incomplete_game_cannot_be_built_as_a_match_log(self):
        config = GameConfig(rows=2, cols=3, win_length=3)
        with self.assertRaises(ValueError):
            _build_match_log(
                game_config=config,
                human_player=1,
                agent_name="Rule",
                agent_seed=27,
                agent_metadata={"kind": "rule"},
                agent_budget=SearchBudget(time_limit_seconds=0.5),
                moves=(),
                result=Game(config).result(),
                output_dir=Path("outputs/ui_games"),
                match_id="unfinished-game",
            )


if __name__ == "__main__":
    unittest.main()
