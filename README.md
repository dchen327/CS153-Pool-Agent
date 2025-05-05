# Pool Agent

## Running Script

- Run `exploration.ipynb` to define board specific constants
- Then run `game.py`, which will launch screen mirroring software and play the game

## File Organzation

- `exploration.ipynb` is run initially to grab game constants (might differ across devices)
- `project.py` has all helper methods used across different files
- `game.py` has all the actual automation, from data collection to actually playing the game
- `label_data.ipynb` is used to label individual balls, which organizes the output into the `labeling/` folder
- `constants.json` stores constants from exploration, but can be manually adjusted

## Setup Notes

- Phone is horizontal, notch to the right
