# TTNTNC-122

Python project for an extended Tic-Tac-Toe game with configurable board size
and winning line length. Players can place a piece in any empty cell.

The game engine is implemented. AI agents, training, evaluation and the console
demo remain scaffolds.

## Requirements

Python 3.10 or later. The engine and its tests use only the standard library.
Currently tested with Python 3.13.1.

## Run engine tests

From this directory:

```powershell
python -B -m unittest discover -s tests -v
```

## Engine example

```powershell
python -B -c "import sys; sys.path.insert(0, 'src'); from ttt_ai.game import Game; game = Game(); game.step(17); print(game.render())"
```

The default configuration is 6 rows, 7 columns and a winning line of at least
4 pieces. Internal actions are zero-based cell indices; rendered row and column
labels are one-based. Invalid moves raise `ValueError` without changing the game.

This repository contains code, technical instructions and tests. Generated
artifacts and unpublished working notes are excluded from version control.
