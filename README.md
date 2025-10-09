Current Progress (as of now)

Arena & Towers

18x32 grid arena loaded from JSON.

Towers modeled as multi-tile structures (4x4 King, 3x3 Princess).

Towers track HP, range, damage, hit speed, and scale with levels (~3.5% per level).

King tower activates when damaged or when a princess tower falls.

Card Data Integration

JSON datasets for troops, spells, buildings, projectiles, and buffs.

Stats per level included (HP, damage, DPS, etc.).

Supports special behaviors (area effects, pushback, spawn units, buffs like Rage/Freeze).

Simulation Core

Troop pathfinding, projectile logic, tower AI implemented.

Shared deploy zones unlocked dynamically after tower destruction.

Matches can be run bot-vs-bot for testing.

Machine Learning Hooks

Environment designed for reinforcement learning (DQN, Q-learning).

Training/evaluation scripts started (CPU & GPU support).

GUI with Tkinter for running matches and tracking results.

Assets

Game images (fight_image.png, arena.png) prepared.

Card datasets matched to in-game stats (RoyaleAPI / extracted game files).

Processed datasets

- `python tools/generate_unit_catalog.py` rebuilds the consolidated unit catalogue in `data/processed/`.
- Columns: `Name`, `Type`, `Elixir Cost`, `Role`, `Counters` (`Counters` stays JSON-formatted so ML loaders can eval/parse quickly).
- Filtering: skips hidden/event/debug exports, arena towers, and zero-cost/training-only cards; any records missing raw stats are reported and omitted until the source data is refreshed.
- Intended use: consistent features for recommendation engines, reward shaping, and deck analytics sitting on top of the simulator.

🚀 Future Progress

Core Features

Implement missing card mechanics (clone, mirror, evolutions, etc.).

Improve troop targeting logic (priority, retarget, kiting).

Add elixir cycle and deck-building system.

AI & Reinforcement Learning

Expand from basic DQN to multi-agent RL.

Train agents across many simulations (parallel matches).

Test AI against rule-based or scripted opponents.

Long-term: achieve real-time decision making (when/where to deploy).

User Experience

More polished GUI (deck selection, match replay, live stats).

Charts/graphs for AI learning progress.

Support for custom matches (human vs bot, bot vs bot, custom decks).

Data & Balance

Keep stats synced with RoyaleAPI (latest balance changes).

Add auto-update pipeline for card data.

Export training results and battle logs.

Advanced Features

Replay system for matches.

Integration with YOLO / Roboflow for troop detection (vision AI).

Long-term: connect to a 1:1 Clash Royale
