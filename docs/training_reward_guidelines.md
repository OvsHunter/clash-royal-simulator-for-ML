# Reinforcement-Learning Training Guidelines

This document outlines practical defaults for training Clash Royale bots with the simulator.

## 1. Reward Shaping

Design dense rewards so the agent receives timely feedback instead of only learning from match outcomes.

| Signal | Suggested Reward | Notes |
| --- | --- | --- |
| Enemy princess tower damage | +0.02 per HP | Scale within ±1 after clipping. |
| Friendly princess tower damage | -0.02 per HP | Symmetric penalty keeps defense relevant. |
| Princess tower destroyed | +100 | Trigger when `Engine.towers` entries flip `alive` to `False`. |
| Friendly princess tower destroyed | -100 | Encourages protecting your side. |
| Enemy king tower destroyed | +300 | Ends the game with a decisive bonus. |
| Friendly king tower destroyed | -300 | Large penalty for being three-crowned. |
| Efficient elixir spend | +0.2 when elixir drops from 10 | Reward avoiding float; check `engine.players[i].elixir`. |
| Elixir overflow | -0.2 each tick at 10 elixir | Penalize wasted generation. |
| Positive elixir trade | +1 per excess elixir value | Compare destroyed troop cost vs. spent card. |
| Negative elixir trade | -1 per deficit elixir value | Keeps trades value-positive. |
| Match victory | +500 | Use `engine.winner` after `Engine.tick`. |
| Match loss | -500 | Mirrors victory reward. |
| Draw (optional) | +100 | Or 0 to discourage draws. |

Clip the cumulative reward to [-1, 1] before pushing transitions into replay buffers if your algorithm is sensitive to large magnitudes.

## 2. State Representation

Expose the agent to the minimum sufficient statistics:

- **Troop positions**: downsample the 18×32 arena grid (`data/arena.json`) into coarse bins and encode ally vs. enemy presence.
- **Elixir**: include current elixir for both players (`engine.players[1].elixir`, `engine.players[2].elixir`).
- **Tower HP**: feed normalized HP for each tower from `engine.towers` and their `hp / hp_max` ratios.
- **Current hand**: one-hot encode the four cards available plus the next card.
- **Timer / overtime flag**: incorporate `engine.time` and the double-/triple-elixir thresholds in `core.rules`.

## 3. Action Space

Discretize the arena so deploying a card becomes choosing among a finite set of zones:

1. For each card in hand, define 8–12 legal deployment zones per lane (back, mid, bridge; left/right).
2. Add a "wait" action for when saving elixir is optimal.
3. Mask illegal actions by checking `Arena.can_deploy(card, tile)` or validating via `Engine.deploy` return values before logging the step.

## 4. Baseline Hyperparameters (DQN-style)

| Parameter | Value |
| --- | --- |
| Discount factor (γ) | 0.99 |
| Learning rate (α) | 1e-4 |
| Replay buffer | 200,000 transitions |
| Batch size | 64 |
| Target update | Every 2,500 gradient steps |
| Exploration (ε) | Start 1.0 → 0.1 over 1M steps |
| Gradient clipping | 1.0 |

Adjust for other algorithms (PPO, A3C) accordingly, but these values provide a stable baseline.

## 5. Real-Time Decision Evaluation

Let rewards, not heuristics, score each action:

1. Apply incremental rewards at every simulation tick to encode tower damage, elixir flow, and troop trades.
2. Log the reward assigned immediately after each action so you can audit whether the signal matches your expectations.
3. Use moving averages over recent rewards to flag actions that consistently perform poorly in similar states.

## 6. Curriculum Strategy

Progressively increase task complexity to avoid overwhelming the agent:

1. **Tower rush drills**: restrict decks to a single win condition per side to learn pathing and tower focus.
2. **Spell introduction**: add direct-damage spells; emphasize timing-based rewards (e.g., splash hits).
3. **Four-card decks**: enable cycling mechanics while keeping state manageable.
4. **Full ladder decks**: unlock all eight cards and both lanes; maintain dense rewards to preserve learnability.

Checkpoint models at each stage and use performance on simpler curricula as regression tests for future tweaks.

## 7. Tooling Tips

- Use `core.simulation.Engine` directly for headless self-play loops; couple it with deck logic in your training script.
- Leverage the GUI (`gui/gui.py`) for qualitative evaluation of learned policies and to inspect tower HP/elixir in real time.
- Store transitions with timestamps so you can reconstruct problematic episodes when tuning rewards.

