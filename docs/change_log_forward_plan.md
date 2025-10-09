# Clash Royale Simulator: Change Log & Forward Plan Prompt

Use this prompt as a comprehensive reference when asking for updates or reviewing past work on the Clash Royale simulator. It distills the repository's current state, notable design decisions, and prioritized next steps so follow-up tasks can be described unambiguously.

## Project Snapshot
- **Arena & Towers**
  - 18x32 tile arena defined in `data/arena.json`; `core/arena.Arena` interprets deploy zones, bridge/river constraints, and dynamically unlocks shared tiles after a princess tower falls.
  - Towers are modeled as multi-tile structures (4x4 king, 3x3 princess) with level-scaling HP/damage (~3.5% per level from tournament baselines). King activation triggers on first damage or allied princess destruction.
- **Troop & Spell Data**
  - `core/troop_data.get_card(name, level)` returns normalized per-level stats from the JSON datasets in `data/`. Projectile, buff, and spawn metadata are already captured for most cards.
- **Simulation Core**
  - `core/simulation.Engine` coordinates deployment, tick updates, pathfinding (`core/pathfinding`), projectile resolution (`core/projectiles`), and effect handling (`core/effects`).
  - Units follow 8-directional A* paths, stop advancing once in range, and apply damage via melee contact or projectile entities. Collision currently lives on a single plane shared by ground and air.
- **GUI & Tooling**
  - Tkinter GUI (`gui/gui.py`) renders the arena, manages manual deployments, and offers quick level overrides for towers/troops.
  - Supplementary tools include the arena editor and CLI diagnostics (`tools/`).
- **Machine-Learning Hooks**
  - Initial reinforcement-learning scaffolding exists in `ai/` and `python/`, with environments geared toward DQN-style experimentation.

## Known Limitations / Outstanding Bugs
1. **Collision layers**: flying units still collide with ground troops because separation planes are not implemented.
2. **Charge & splash mechanics**: cards with charges, spawn-impact damage, or multi-target splash rely on placeholder logic.
3. **Spell/buff durations**: effects such as Rage, Freeze, and damage-over-time need precise tick-level modeling.
4. **Deck management**: there is no elixir cycle, hand rotation, or card cooldown enforcement in the GUI sandbox.
5. **Data freshness**: stats match RoyaleAPI exports from early 2025; balance patches will require updated JSON conversions.

## Completed Changes (High-Level)
- Baseline arena/tower implementation with level scaling and activation rules.
- Per-card stat ingestion from RoyaleAPI with normalized targeting and projectile definitions.
- Pathfinding with river/bridge constraints and per-tick projectile resolution.
- Tkinter sandbox for manual deployments, including tower level controls and visual HUD.
- Documentation of engine architecture and datasets in `README.md` and `docs/project_knowledge.md`.

## Priority Roadmap
### 1. Combat Fidelity
- Implement charge tracking (Prince, Battle Ram, etc.) with acceleration, impact damage, and reset logic.
- Model spawn-impact effects (Mega Knight, Electro Wizard, spirits) with area-of-effect resolution and stagger.
- Add splash radius metadata to troop JSON and update damage loops for true multi-target hits.

### 2. Collision & Targeting Refinements
- Separate collision planes for ground vs. flying entities while preserving targeting interactions.
- Improve retargeting when preferred targets are eliminated or move out of range (kiting behaviors).
- Handle shield mechanics (e.g., Dark Prince) and companion units (Goblin Giant).

### 3. Economy & Deck Systems
- Add elixir regeneration tied to match time (double/triple elixir phases already tracked on the HUD).
- Implement four-card hand rotation, card queue, and constraints on redeploying the same card.
- Expose deck selection UI and validation for bot and human-controlled matches.

### 4. AI & Automation
- Expand RL environment wrappers to handle full deck/economy rules.
- Provide scripted opponents for benchmarking (rule-based heuristics).
- Add logging hooks for match telemetry and reward shaping.

### 5. Data & Tooling
- Build converters for the latest RoyaleAPI exports and establish a balance update pipeline.
- Document card-specific ability quirks in a dedicated reference file.
- Enhance arena editor with collision/zoning overlays and export validation.

## Suggested Prompt Template for Future Tasks
```
You are working on the Clash Royale simulator.
Current functionality: arena/tower logic, troop data ingestion, pathfinding, projectile system, and Tkinter GUI sandbox (see docs/detailed_change_plan.md for details).
Outstanding gaps: collision layers, charge/spawn mechanics, deck & elixir systems, ML integration polish, data refresh pipeline.

Task:
- [Clearly describe the feature/bug to address.]
- Reference relevant modules (core/simulation.py, core/troop_data.py, gui/gui.py, etc.).
- Define acceptance criteria (e.g., unit tests in tests/, GUI behavior, documentation updates).
- Include testing expectations (run pytest, integration script, etc.).
```

Keep this document updated whenever new mechanics are implemented, data sources change, or priorities shift so collaborators always have a current snapshot of past changes and planned work.
