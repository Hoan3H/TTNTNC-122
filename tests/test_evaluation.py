import json
from dataclasses import replace
from datetime import date
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt_ai.evaluation import (
    AgentEntry,
    EvaluationConfig,
    MatchLog,
    play_match,
    load_logs,
    save_logs,
)
from ttt_ai.game import (
    AgentDecision,
    GameConfig,
    SearchStats,
)


class ScriptAgent:
    def __init__(self, game_config, actions):
        self.game_config = game_config
        self.actions = iter(actions)

    def select_action(self, state, budget=None):
        return AgentDecision(
            action=next(self.actions),
            stats=SearchStats(elapsed_seconds=0.01, nodes_visited=1),
        )


def entry(name, config, actions, metadata=None):
    return AgentEntry(
        name=name,
        factory=lambda seed: ScriptAgent(config, actions),
        metadata=metadata or {"source": name},
    )


def winning_match(opening=()):
    config = GameConfig(2, 3, 3)
    return play_match(
        config,
        entry("script-x", config, (0, 1, 2), {"version": 1}),
        entry("script-o", config, (3, 4, 5), {"version": 2}),
        EvaluationConfig(games_per_pair=1, output_dir=Path("tmp-evaluation")),
        match_id="match-1",
        block_id="block-1",
        seed_x=11,
        seed_o=12,
        opening=opening,
    )


class EvaluationTests(unittest.TestCase):
    def test_play_match_records_each_agent_turn_and_reaches_terminal_result(self):
        log = winning_match(opening=(2,))
        self.assertIsInstance(log, MatchLog)
        self.assertEqual(log.opening, (2,))
        self.assertEqual(log.agent_x, "script-x")
        self.assertEqual(log.agent_o, "script-o")
        self.assertEqual(log.seed_x, 11)
        self.assertEqual(log.seed_o, 12)
        self.assertEqual(log.agent_metadata["X"]["agent"], {"version": 1})
        self.assertIsNone(log.agent_metadata["X"]["budget"])
        self.assertEqual(log.agent_metadata["O"]["agent"], {"version": 2})
        self.assertIsNone(log.agent_metadata["O"]["budget"])
        self.assertEqual(tuple(move.action for move in log.moves), (3, 0, 4, 1))
        self.assertEqual(tuple(move.state.to_play for move in log.moves), (-1, 1, -1, 1))
        self.assertEqual(log.result.winner, 1)
        self.assertTrue(log.result.terminal)

    def test_play_match_rejects_illegal_agent_action(self):
        config = GameConfig(2, 3, 3)
        illegal_x = entry("illegal", config, (True,))
        normal_o = entry("normal", config, (3,))
        with self.assertRaises(ValueError):
            play_match(
                config,
                illegal_x,
                normal_o,
                EvaluationConfig(games_per_pair=1),
                match_id="bad-match",
                block_id="block",
                seed_x=1,
                seed_o=2,
            )

    def test_play_match_rejects_illegal_opening(self):
        config = GameConfig(2, 3, 3)
        with self.assertRaises(ValueError):
            play_match(
                config,
                entry("x", config, (0,)),
                entry("o", config, (1,)),
                EvaluationConfig(games_per_pair=1),
                match_id="bad-opening",
                block_id="block",
                seed_x=1,
                seed_o=2,
                opening=(True,),
            )

    def test_save_and_load_round_trip_match_logs(self):
        log = winning_match()
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            save_logs((log,), output_dir)
            loaded = load_logs(output_dir)
            self.assertEqual(loaded, (log,))
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["match_count"], 1)
            self.assertEqual(manifest["schema_version"], 1)

    def test_empty_log_set_round_trips(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "empty"
            save_logs((), output_dir)
            self.assertEqual(load_logs(output_dir), ())

    def test_date_scoped_logs_use_separate_files_and_round_trip(self):
        first = winning_match()
        second = replace(first, match_id="match-2")
        next_day = replace(first, match_id="match-3")
        first_day = date(2026, 11, 17)
        second_day = date(2026, 11, 18)

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            save_logs((first,), output_dir, day=first_day)
            save_logs(
                load_logs(output_dir, day=first_day) + (second,),
                output_dir,
                day=first_day,
            )
            save_logs((next_day,), output_dir, day=second_day)

            self.assertEqual(
                load_logs(output_dir, day=first_day), (first, second)
            )
            self.assertEqual(load_logs(output_dir, day=second_day), (next_day,))
            self.assertTrue(
                (output_dir / "matches_2026-11-17.jsonl").is_file()
            )
            self.assertTrue(
                (output_dir / "manifest_2026-11-18.json").is_file()
            )

    def test_date_scoped_logs_reject_non_date_values(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_logs((), Path(temp_dir), day="2026-11-17")

    def test_load_logs_rejects_action_that_does_not_replay(self):
        log = winning_match()
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            save_logs((log,), output_dir)
            record = json.loads(
                (output_dir / "matches.jsonl").read_text(encoding="utf-8")
            )
            record["moves"][0]["action"] = 99
            (output_dir / "matches.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_logs(output_dir)

    def test_load_logs_checks_manifest_count_and_schema(self):
        log = winning_match()
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            save_logs((log,), output_dir)
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["match_count"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_logs(output_dir)

            manifest["schema_version"] = 99
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_logs(output_dir)

    def test_save_rejects_duplicate_ids_and_non_json_metadata(self):
        log = winning_match()
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            with self.assertRaises(ValueError):
                save_logs((log, log), output_dir)
            self.assertFalse(output_dir.exists())

            bad_log = winning_match()
            bad_log.agent_metadata["X"]["not_json"] = object()
            with self.assertRaises(TypeError):
                save_logs((bad_log,), output_dir)
            self.assertFalse(output_dir.exists())

    def test_load_logs_rejects_duplicate_match_ids(self):
        log = winning_match()
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "logs"
            save_logs((log,), output_dir)
            matches_path = output_dir / "matches.jsonl"
            first_record = json.loads(matches_path.read_text(encoding="utf-8"))
            matches_path.write_text(
                json.dumps(first_record) + "\n" + json.dumps(first_record) + "\n",
                encoding="utf-8",
            )
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["match_count"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_logs(output_dir)


if __name__ == "__main__":
    unittest.main()
