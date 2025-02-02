# Croissant Game Research Project

This repo was created for a research project. It contains a simple text game and scripts to play the game via AI models in two different ways. See [the blog post](https://bunnybaxter.substack.com/p/ai-invests-its-hard-earned-money) for more about the research and results.

Feel free to contact me with questions.

## The Croissant Game

This simple game involves investing money to buy croissants. There are two versions: the regular one, and one with a "stash" function that allows you to save money for future games.

If you would like to play, examine, or modify the game itself, there are three relevant files:
* `play_console.py`, which contains the console interface for the game. Run it with no arguments to play the regular version. Run with the `--enable_stash` flag for the version with the stash enabled.
* `game_model.py`, which contains model code for running the game, but no interface. This file is shared with the AI scripts.
* `game_config.toml`, which contains all tweakable numbers for the game in a convenient config file.

## Training an RL model with PPO

The first method of playing with AI uses PPO to train a Reinforcement Learning model on the Croissant Game.

Required libraries: `torch torchrl tensordict gymnasium<1.0 numpy`.

Run `ai_train.py` to train a model. It will take its hyperparameters and whether to enable the stash from `hyperparams.toml`. It will save the model checkpoint to the checkpoints/ folder using `torch.save`. The gymnasium environment for the game is in `game_env.py`.

To run a previously-trained model, use `ai_play.py` and specify the checkpoint file with the `--model` flag.
* Use the flag `--enable_stash` to play the game with the stash enabled. This must match the setting the model was trained with.
* Use the flag `--iterations` to specify how many games the model should play in a row.
* Use the flag `--deterministic` if you would like the model to always play the action with the highest probability, rather than sampling from the action probabilities.

## Prompting Claude to play the game

The second method of playing with AI uses the Anthropic API to prompt Claude for an action with the game state. It uses the specific model `claude-3-5-sonnet-20241022`.

Required libraries: `anthropic`.

All code for this method is in `claude_play.py`. By default, it calls a fake API defined in `fake_anthropic_api.py` for testing purposes. Use the `--real_api` flag to send messages to the Anthropic server. The API key is read from environment variables.

The message log, final score, and temperature parameter are saved as JSON to the logs/ folder.

Claude currently only plays the regular version of the game. The game aborts if Claude returns an illegal or invalid action (a log file is still written). Enable chain-of-thought prompting with the `--thoughts` flag.
