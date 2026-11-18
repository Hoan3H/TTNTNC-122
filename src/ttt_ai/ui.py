from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import secrets
from tkinter import filedialog, messagebox, StringVar, Text, Tk, ttk
import tkinter as tk
from uuid import uuid4

from .agents import Agent, MinimaxAgent, RandomAgent, RuleAgent
from .dqn import DQNAgent
from .evaluation import (
    EvaluationConfig,
    MatchLog,
    MoveLog,
    load_logs,
    save_logs,
)
from .game import (
    AgentDecision,
    Game,
    GameConfig,
    GameResult,
    Player,
    SearchBudget,
    SearchStats,
)


def _build_match_log(
    game_config: GameConfig,
    human_player: Player,
    agent_name: str,
    agent_seed: int,
    agent_metadata: dict[str, object],
    agent_budget: SearchBudget | None,
    moves: tuple[MoveLog, ...],
    result: GameResult,
    output_dir: Path,
    match_id: str,
) -> MatchLog:
    if type(human_player) is not int or human_player not in (-1, 1):
        raise ValueError("human_player must be -1 or 1.")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name must be a non-empty string.")
    if type(agent_seed) is not int:
        raise ValueError("agent_seed must be an integer.")
    if not isinstance(agent_metadata, dict):
        raise ValueError("agent_metadata must be a dictionary.")
    if agent_budget is not None and not isinstance(agent_budget, SearchBudget):
        raise ValueError("agent_budget must be a SearchBudget or None.")
    if not isinstance(moves, tuple) or any(
        not isinstance(move, MoveLog) for move in moves
    ):
        raise ValueError("moves must be a tuple of MoveLog objects.")
    if not isinstance(result, GameResult) or not result.terminal:
        raise ValueError("A completed game result is required.")
    if not isinstance(output_dir, Path):
        raise ValueError("output_dir must be a Path.")
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("match_id must be a non-empty string.")

    agent_by_player = {
        human_player: "Human",
        -human_player: agent_name,
    }
    metadata_by_player = {
        human_player: {"agent": {"kind": "human"}, "budget": None},
        -human_player: {
            "agent": dict(agent_metadata),
            "budget": (
                {
                    "max_depth": agent_budget.max_depth,
                    "max_simulations": agent_budget.max_simulations,
                    "time_limit_seconds": agent_budget.time_limit_seconds,
                }
                if agent_budget is not None
                else None
            ),
        },
    }
    seed_by_player = {human_player: 0, -human_player: agent_seed}
    protocol = EvaluationConfig(
        games_per_pair=1,
        seeds=(agent_seed,),
        split="interactive",
        output_dir=output_dir,
    )
    return MatchLog(
        match_id=match_id,
        block_id="interactive-ui",
        game_config=game_config,
        protocol=protocol,
        agent_x=agent_by_player[1],
        agent_o=agent_by_player[-1],
        seed_x=seed_by_player[1],
        seed_o=seed_by_player[-1],
        opening=(),
        agent_metadata={
            "X": metadata_by_player[1],
            "O": metadata_by_player[-1],
        },
        moves=moves,
        result=result,
    )


class GameUI:
    def __init__(
        self,
        root: Tk,
        game_config: GameConfig | None = None,
        output_dir: Path = Path("outputs/ui_games"),
    ) -> None:
        if not isinstance(root, Tk):
            raise ValueError("root must be a Tk window.")
        if game_config is None:
            game_config = GameConfig()
        Game(game_config)
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a Path.")

        self.root = root
        self.game_config = game_config
        self.output_dir = output_dir
        self.game = Game(game_config)
        self.human_player: Player = 1
        self.agent: Agent | None = None
        self.agent_name = ""
        self.agent_seed = 0
        self.agent_metadata: dict[str, object] = {}
        self.agent_budget: SearchBudget | None = None
        self.dqn_checkpoint: Path | None = None
        self.moves: list[MoveLog] = []
        self.match_id = ""
        self.game_active = False
        self.match_saved = False
        self._generation = 0

        self.agent_choice = StringVar(value="Rule")
        self.human_side = tk.IntVar(value=1)
        self.status = StringVar(value="Choose an agent and side, then start a new game.")
        self.checkpoint_label = StringVar(value="No DQN checkpoint selected")

        self.root.title("Tic-Tac-Toe AI")
        self.root.minsize(680, 520)
        self._build_widgets()
        self._refresh_board()
        self._refresh_log()

    def _build_widgets(self) -> None:
        navigation = ttk.Frame(self.root, padding=(12, 12, 12, 4))
        navigation.pack(fill="x")
        ttk.Button(
            navigation, text="Game", command=self._show_game
        ).pack(side="left")
        ttk.Button(
            navigation, text="Current Game Log", command=self._show_log
        ).pack(side="left", padx=(8, 0))

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.play_screen = ttk.Frame(container)
        self.log_screen = ttk.Frame(container)
        self.play_screen.grid(row=0, column=0, sticky="nsew")
        self.log_screen.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(self.play_screen)
        controls.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Agent:").pack(side="left")
        self.agent_selector = ttk.Combobox(
            controls,
            textvariable=self.agent_choice,
            values=("Random", "Rule", "Minimax", "DQN"),
            state="readonly",
            width=12,
        )
        self.agent_selector.pack(side="left", padx=(6, 12))
        self.agent_selector.bind("<<ComboboxSelected>>", self._on_agent_changed)

        ttk.Label(controls, text="You play:").pack(side="left")
        self.x_radio = ttk.Radiobutton(
            controls,
            text="X",
            value=1,
            variable=self.human_side,
        )
        self.x_radio.pack(side="left", padx=(6, 0))
        self.o_radio = ttk.Radiobutton(
            controls,
            text="O",
            value=-1,
            variable=self.human_side,
        )
        self.o_radio.pack(side="left", padx=(2, 12))
        self.new_game_button = ttk.Button(
            controls, text="Start New Game", command=self._start_new_game
        )
        self.new_game_button.pack(side="left")

        self.checkpoint_button = ttk.Button(
            self.play_screen,
            text="Choose DQN Checkpoint",
            command=self._choose_checkpoint,
        )
        self.checkpoint_button.pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.play_screen,
            textvariable=self.checkpoint_label,
            wraplength=620,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            self.play_screen, textvariable=self.status, font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))
        self.board_frame = ttk.Frame(self.play_screen)
        self.board_frame.pack(anchor="center")
        self.board_buttons: list[list[tk.Button]] = []
        for row in range(self.game_config.rows):
            ttk.Label(self.board_frame, text=str(row + 1), width=3).grid(
                row=row + 1, column=0, padx=2, pady=2
            )
            button_row = []
            for col in range(self.game_config.cols):
                action = row * self.game_config.cols + col
                button = tk.Button(
                    self.board_frame,
                    text="",
                    width=4,
                    height=2,
                    font=("Segoe UI", 14, "bold"),
                    command=lambda selected=action: self._on_cell_clicked(selected),
                )
                button.grid(
                    row=row + 1,
                    column=col + 1,
                    padx=2,
                    pady=2,
                    sticky="nsew",
                )
                button_row.append(button)
            self.board_buttons.append(button_row)
        ttk.Label(self.board_frame, text="", width=3).grid(row=0, column=0)
        for col in range(self.game_config.cols):
            ttk.Label(self.board_frame, text=str(col + 1), width=4).grid(
                row=0, column=col + 1, padx=2
            )

        ttk.Label(
            self.play_screen,
            text=f"Logs saved to: {self.output_dir}",
            wraplength=620,
        ).pack(anchor="w", pady=(12, 0))

        ttk.Label(
            self.log_screen,
            text="Moves in the current game",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        self.log_text = Text(self.log_screen, height=20, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        self.log_status = StringVar(value="No game has been played.")
        ttk.Label(
            self.log_screen, textvariable=self.log_status, wraplength=620
        ).pack(anchor="w", pady=(8, 0))

        self._on_agent_changed()
        self._show_game()

    def _on_agent_changed(self, _event: object = None) -> None:
        is_dqn = self.agent_choice.get() == "DQN"
        self.checkpoint_button.configure(
            state="normal" if is_dqn and not self.game_active else "disabled"
        )
        if not is_dqn:
            self.checkpoint_label.set("A checkpoint is only needed for DQN.")
        elif self.dqn_checkpoint is not None:
            self.checkpoint_label.set(str(self.dqn_checkpoint))

    def _choose_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose DQN Checkpoint",
            filetypes=(("PyTorch checkpoints", "*.pt"), ("All files", "*.*")),
        )
        if selected:
            self.dqn_checkpoint = Path(selected)
            self.checkpoint_label.set(str(self.dqn_checkpoint))

    def _start_new_game(self) -> None:
        if not self.game.is_terminal() and self.moves:
            if not messagebox.askyesno(
                "Start a new game?",
                "The current game is unfinished and will not be saved. Continue?",
                parent=self.root,
            ):
                return

        selected_agent = self.agent_choice.get()
        seed = secrets.randbits(32)
        try:
            agent, metadata, budget = self._create_agent(selected_agent, seed)
        except (
            EOFError,
            ImportError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            messagebox.showerror(
                "Could not create agent",
                str(error),
                parent=self.root,
            )
            return

        self._generation += 1
        self.game = Game(self.game_config)
        self.human_player = self.human_side.get()
        self.agent = agent
        self.agent_name = selected_agent
        self.agent_seed = seed
        self.agent_metadata = metadata
        self.agent_budget = budget
        self.moves = []
        self.match_id = uuid4().hex
        self.game_active = True
        self.match_saved = False
        self.agent_selector.configure(state="disabled")
        self.x_radio.configure(state="disabled")
        self.o_radio.configure(state="disabled")
        self.new_game_button.configure(text="Start Another Game")
        self._on_agent_changed()
        self._refresh_board()
        self._refresh_log()
        self._update_status()
        if self.game.snapshot().to_play != self.human_player:
            generation = self._generation
            self.root.after(100, lambda: self._play_agent_turn(generation))

    def _create_agent(
        self, name: str, seed: int
    ) -> tuple[Agent, dict[str, object], SearchBudget | None]:
        if name == "Random":
            return RandomAgent(self.game_config, seed), {"kind": "random"}, None
        if name == "Rule":
            return RuleAgent(self.game_config, seed), {"kind": "rule"}, None
        if name == "Minimax":
            budget = SearchBudget(time_limit_seconds=0.5)
            return (
                MinimaxAgent(self.game_config, depth=8, seed=seed),
                {"kind": "minimax", "max_depth": 8},
                budget,
            )
        if name == "DQN":
            if self.dqn_checkpoint is None:
                raise ValueError("Select a trained DQN checkpoint first.")
            agent = DQNAgent(self.game_config, seed=seed, device="cpu")
            agent.load_checkpoint(self.dqn_checkpoint)
            if not isinstance(agent.metadata.get("training"), dict):
                raise ValueError(
                    "The checkpoint has no training metadata. Select a checkpoint "
                    "created by DQNAgent.train()."
                )
            return (
                agent,
                {
                    "kind": "dqn",
                    "checkpoint": self.dqn_checkpoint.name,
                    "encoding_version": agent.config.encoding_version,
                },
                None,
            )
        raise ValueError(f"Unsupported agent: {name}")

    def _on_cell_clicked(self, action: int) -> None:
        if (
            not self.game_active
            or self.game.is_terminal()
            or self.game.snapshot().to_play != self.human_player
            or action not in self.game.legal_actions()
        ):
            return
        try:
            self._apply_action(action, SearchStats(elapsed_seconds=0.0))
        except (TypeError, ValueError, RuntimeError) as error:
            messagebox.showerror("Invalid move", str(error), parent=self.root)

    def _play_agent_turn(self, generation: int) -> None:
        if (
            generation != self._generation
            or not self.game_active
            or self.agent is None
            or self.game.is_terminal()
            or self.game.snapshot().to_play == self.human_player
        ):
            return
        self._refresh_board()
        self.status.set(f"{self.agent_name} is thinking...")
        try:
            decision = self.agent.select_action(
                self.game.snapshot(), self.agent_budget
            )
            if not isinstance(decision, AgentDecision):
                raise TypeError("Agent must return an AgentDecision.")
            if not isinstance(decision.stats, SearchStats):
                raise TypeError("Agent must return valid SearchStats.")
            if (
                type(decision.action) is not int
                or decision.action not in self.game.legal_actions()
            ):
                raise ValueError("Agent returned an illegal move.")
            self._apply_action(decision.action, decision.stats)
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
            self.game_active = False
            self.agent_selector.configure(state="readonly")
            self.x_radio.configure(state="normal")
            self.o_radio.configure(state="normal")
            self._on_agent_changed()
            self.status.set(f"{self.agent_name} could not choose a move.")
            messagebox.showerror(
                "Agent Error",
                f"{error}\n\nChoose another agent and start a new game.",
                parent=self.root,
            )
        self._refresh_board()

    def _apply_action(self, action: int, stats: SearchStats) -> None:
        state = self.game.snapshot()
        if action not in self.game.legal_actions():
            raise ValueError("The cell is occupied or the game is over.")
        self.game.step(action)
        self.moves.append(MoveLog(state=state, action=action, stats=stats))
        self._refresh_board()
        self._refresh_log()
        if self.game.is_terminal():
            self._finish_game()
        elif self.game.snapshot().to_play != self.human_player:
            self.status.set(f"{self.agent_name} is up next...")
            generation = self._generation
            self.root.after(100, lambda: self._play_agent_turn(generation))
        else:
            self._update_status()

    def _finish_game(self) -> None:
        result = self.game.result()
        if result.winner == self.human_player:
            self.status.set("You win!")
        elif result.winner is None:
            self.status.set("Draw.")
        else:
            self.status.set(f"{self.agent_name} wins.")

        self.game_active = False
        self.agent_selector.configure(state="readonly")
        self.x_radio.configure(state="normal")
        self.o_radio.configure(state="normal")
        self._on_agent_changed()
        try:
            self._save_completed_game()
        except (OSError, TypeError, ValueError) as error:
            self.match_saved = False
            self.log_status.set(f"Could not save game: {error}")
            messagebox.showerror(
                "Could Not Save Game",
                f"The game is over but could not be saved:\n{error}",
                parent=self.root,
            )
        self._refresh_board()
        self._refresh_log()

    def _save_completed_game(self) -> None:
        day = datetime.now().astimezone().date()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / f"manifest_{day.isoformat()}.json"
        matches_path = self.output_dir / f"matches_{day.isoformat()}.jsonl"
        if manifest_path.exists() != matches_path.exists():
            raise ValueError("The current day's log file pair is incomplete.")
        existing_logs = (
            load_logs(self.output_dir, day=day) if manifest_path.exists() else ()
        )
        match = _build_match_log(
            game_config=self.game_config,
            human_player=self.human_player,
            agent_name=self.agent_name,
            agent_seed=self.agent_seed,
            agent_metadata=self.agent_metadata,
            agent_budget=self.agent_budget,
            moves=tuple(self.moves),
            result=self.game.result(),
            output_dir=self.output_dir,
            match_id=self.match_id,
        )
        save_logs(existing_logs + (match,), self.output_dir, day=day)
        self.match_saved = True
        self.log_status.set(f"Saved: {matches_path.name}")

    def _refresh_board(self) -> None:
        state = self.game.snapshot()
        symbols = {0: "", 1: "X", -1: "O"}
        can_human_move = (
            self.game_active
            and not self.game.is_terminal()
            and state.to_play == self.human_player
        )
        for row, cells in enumerate(state.board):
            for col, cell in enumerate(cells):
                button = self.board_buttons[row][col]
                button.configure(
                    text=symbols[cell],
                    state=(
                        "normal"
                        if can_human_move and cell == 0
                        else "disabled"
                    ),
                    disabledforeground="#202020",
                )

    def _refresh_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        if not self.moves:
            self.log_text.insert("end", "No moves in the current game yet.\n")
        else:
            for index, move in enumerate(self.moves, start=1):
                row, col = divmod(move.action, self.game_config.cols)
                actor = (
                    "You"
                    if move.state.to_play == self.human_player
                    else self.agent_name
                )
                line = f"{index:02}. {actor}: row {row + 1}, column {col + 1}"
                if actor != "You":
                    line += f" - {move.stats.elapsed_seconds:.3f}s"
                self.log_text.insert("end", line + "\n")
        if self.game.is_terminal():
            result = self.game.result()
            if result.winner is None:
                self.log_text.insert("end", "\nResult: Draw\n")
            else:
                winner = "You" if result.winner == self.human_player else self.agent_name
                self.log_text.insert("end", f"\nResult: {winner} wins\n")
        self.log_text.configure(state="disabled")

    def _update_status(self) -> None:
        if not self.game_active:
            self.status.set("Choose an agent and side, then start a new game.")
            return
        if self.game.is_terminal():
            return
        current_player = self.game.snapshot().to_play
        if current_player == self.human_player:
            self.status.set(f"Your turn ({'X' if current_player == 1 else 'O'}).")
        else:
            self.status.set(f"{self.agent_name}'s turn.")

    def _show_game(self) -> None:
        self.play_screen.tkraise()

    def _show_log(self) -> None:
        self.log_screen.tkraise()


def main() -> None:
    root = Tk()
    GameUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
