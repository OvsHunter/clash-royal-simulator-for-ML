# Clash Royale Simulator - Project Knowledge

## Project Goal
- Deliver a lightweight but accurate Clash Royale sandbox that supports real card stats, authentic arena rules, and AI/automation hooks.
- Target feature parity for troop/tower/spell logic, deploy restrictions, bot-vs-bot play, and reinforcement-learning pipelines.

---

## Implementation Snapshot (April 2025)
- **Core arena**: `data/arena.json` (18x32 grid) with labeled zones; `core/arena.Arena` loads it, tracks tower footprints, conditional zones, and unlocks shared deploy tiles when a princess tower falls. Damaging any princess tower now wakes the allied king tower automatically.
- **Tower model**: towers are treated as single multi-tile structures (4x4 king, 3x3 princess). They expose HP, range, damage, hit speed, active flag, and level metadata. `Engine.set_tower_levels` lets us scale HP/DMG (about 3.5% per level delta from level 11); GUI spinboxes hook into this.
- **Troop data**: `core/troop_data.get_card(name, level)` returns per-level stats derived from `data/troops.json` (hp/dmg arrays, radius, projectile metadata, flying flag, targets). Default level respects tournament standards (Common 11, Rare 9, Epic 6, Legendary/Champion 5).
- **Simulation** (`core/simulation.Engine`):
  - 8-way A* pathfinding with octile heuristic and river/bridge constraints (`core/pathfinding`).
  - Units store projectile attributes and stop advancing once targets are inside range. Ranged attacks spawn projectiles; melee units apply damage directly.
  - Collisions currently prevent enemy bodies from overlapping; all units share a single collision plane (flying vs ground separation is on the roadmap).
  - Tower and spell projectiles update every tick; spawn levels applied per deploy call.
- **GUI** (`gui/gui.py`): canvas renderer plus side panel controls for card selection, troop level, tower level, and an "Apply Levels" button that pushes values into the engine. HUD shows time, elixir, crowns, and elixir phase.

---

## Arena & Zones
- Dimensions: width 18, height 32.
- Tile labels drive logic:
  - `deploy_p1*` / `deploy_p2*` for base deploy tiles (including conditional lanes).
  - `river` blocks ground units except on `bridge` tiles; flying units ignore the restriction.
  - `unplayable` cannot be traversed or deployed on.
- Conditional tiles (for example `deploy_p2_cond_left`) belong to the owning player until their matching princess tower dies, then open for both players.

---

## Towers
- King towers start inactive; a single hit (or an allied princess falling) activates them permanently.
- Princess towers only target their half of the map: P1 targets rows south of the river, P2 targets north.
- Towers maintain HP percentage, damage, and cooldown when their level changes; damage modifiers (crown tower) stay intact.

See [`docs/training_reward_guidelines.md`](docs/training_reward_guidelines.md) for practical reward shaping, state encoding, and hyperparameter defaults when training bots on top of the simulator.

---

## Troop Engine
- Deployment uses per-player level defaults; callers can override per card via `Engine.deploy(..., level=...)`.
- Units follow paths tile by tile, slow to stay in formation, and hold position once their target is in range.
- Attack loop respects `hit_speed`, `range`, projectile radius, flying/ground targeting, and HP bars update accordingly.
- Current limitations / TODOs:
  - Collision does not yet separate flying vs ground planes (flying units still collide with ground bodies).
  - Charge mechanics, splash damage, spawn-impact effects, and companion units use placeholder behavior until bespoke data is wired in.

---

## Data Pipeline
- Source JSON: `data/troops.json`, `data/spells.json`, `data/projectiles.json`, and related files converted from RoyaleAPI exports.
- `core/troop_data` normalizes targeting strings, collision radii, projectile stats, and per-level arrays; missing values fall back to `core/rules` constants.
- Future work: enrich JSON with flags for splash radius, spawn effects, charge timings, crowd-control durations, and companion unit templates.

---

## GUI & Tooling
- `gui/gui.py`: manual spawn sandbox with level controls; updates simulation time-step at roughly 30 FPS.
- `gui/arena_editor.py`: visual tile editor with color-coded zones.
- CLI helpers in `tools/` for quick testing (`find_target_test`, etc.).

---

## Upcoming Work
1. **Troop fidelity**: add charge logic (Prince, Dark Prince, Battle Ram, Ram Rider), spawn impact effects (Mega Knight, Electro Wizard, spirits), splash damage metadata, crowd-control interactions, and companion units (Goblin Giant backpack).
2. **Collision layers**: decouple flying vs ground collisions while preserving attack interactions.
3. **Deck and elixir systems** (Stage 6): implement hand rotation, elixir regen pacing, deployment legality, and GUI readouts.
4. **Advanced behaviors**: spell damage-over-time/buffs, bot heuristics, headless sims, reinforcement-learning hooks, diagnostics overlays.
5. **Data documentation**: capture troop-specific ability notes in a dedicated reference file to align simulator logic with live-game expectations.

_Keep this document in sync as features land so everyone shares the same mental model of the current simulator state and the remaining roadmap._
