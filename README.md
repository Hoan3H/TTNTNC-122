# TTNTNC-122

Python project for an extended Tic-Tac-Toe game with configurable board size
and winning line length. Players can place a piece in any empty cell.

The game engine, baseline Random/Rule/Minimax agents, single-match logging,
DQN training pipeline and a local Tkinter game UI are implemented. MCTS,
tournament analysis and the console demo remain scaffolds.

## Requirements

Python 3.10 or later. The game engine uses the standard library; DQN and the
full test suite require CPU PyTorch.

Install the DQN dependency from this directory:

```powershell
python -m pip install -r requirements.txt
```

Currently tested with Python 3.13.1 and CPU PyTorch.

## Run tests

From this directory:

```powershell
python -B -m unittest discover -s tests -v
```

## Play with the desktop UI

From this directory, run:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from ttt_ai.ui import main; main()"
```

Choose Random, Rule or Minimax and click cells to play. DQN is available only
after selecting a checkpoint produced by `DQNAgent.train()`. Completed games are
saved by local date under `outputs/ui_games/` as
`matches_YYYY-MM-DD.jsonl` with a matching `manifest_YYYY-MM-DD.json`.

## Engine example

```powershell
python -B -c "import sys; sys.path.insert(0, 'src'); from ttt_ai.game import Game; game = Game(); game.step(17); print(game.render())"
```

The default configuration is 6 rows, 7 columns and a winning line of at least
4 pieces. Internal actions are zero-based cell indices; rendered row and column
labels are one-based. Invalid moves raise `ValueError` without changing the game.

This repository contains code, technical instructions and tests. Generated
artifacts and unpublished working notes are excluded from version control.
