# Troop Mechanics Completion Blueprint

This blueprint expands the earlier high-level playbook into a concrete, end-to-end delivery plan for troop mechanics. It combines audit tasks, data-model upgrades, engine milestones, QA tooling, and documentation hygiene so the simulator reaches feature parity with the live game without creating regressions.

## 1. Establish a Shared Baseline

1. **Inventory the current implementation**
   - Summarize what already exists for every troop archetype (chargers, dashers, divers, spirits, tanks, swarms) by scraping `data/troops.json` and recording which flags the engine already consumes. Capture the gaps in a living table inside this file so contributors know exactly which mechanics are still missing.
   - Re-read the existing unit model before proposing changes. `core/simulation.Unit` already carries charge, splash, spawn, slow, chain, and companion hooks that we can extend instead of rewriting from scratch. 【F:core/simulation.py†L42-L96】
   - Confirm what the data loader exposes. `core/troop_data.get_card` normalizes targets, projectile metadata, and tournament levels but still lacks structured ability definitions. 【F:core/troop_data.py†L68-L208】
   - Align the scope with the project roadmap so efforts fold into the simulator vision. `docs/project_knowledge.md` calls out troop fidelity work (charge, spawn impacts, splash, crowd control, companions) as the number-one upcoming task. 【F:docs/project_knowledge.md†L39-L68】

2. **Create regression anchors**
   - Build at least one deterministic scenario per mechanic using the existing sandbox tests or lightweight scripts in `tools/`. Capture current behavior (even if incorrect) so future diffs can be reviewed against known baselines.
   - Snapshot replay data and GUI recordings for complex interactions (e.g., Mega Knight jump) to speed up manual QA later.

## 2. Formalize the Data Model

1. **Schema enhancements**
   - Extend `data/troops.json` (and optional overrides) with explicit sections:
     ```json
     {
       "charge": {"windup": 2.5, "speed_mult": 1.6, "damage_mult": 2.0, "reset_on_hit": true},
       "dash": {"range": 3.5, "impact_radius": 1.2, "stun": 0.5},
       "splash": {"radius": 1.5, "falloff": 0.35, "chain_max": 3},
       "spawn_effect": {"delay": 0.2, "radius": 3.0, "damage": 290, "stun": 0.5},
       "companions": [{"card": "Spear Goblin", "count": 2, "attach_offset": [0.3, -0.2]}]
     }
     ```
   - Add enum-like strings (`"charge_type": "ramp" | "dash" | "leap"`) so unusual cards (Ram Rider, Bandit) can share code paths without bespoke booleans.
   - Version the schema inside the JSON (`"mechanics_schema": 1`) so future migrations are traceable.

2. **Normalization updates**
   - Expand `_normalize_card` to deep-copy the mechanic dictionaries, apply defaults, and strip zeroed values to keep payloads lean. 【F:core/troop_data.py†L138-L176】
   - Teach `_inject_projectile_metadata` to fill `chain_targets`, `projectile_spawn_effect`, and other projectile-specific mechanics when present. 【F:core/troop_data.py†L193-L208】
   - Record tournament default levels next to mechanic payloads so downstream systems can scale effects per card level without recomputing lookups. 【F:core/troop_data.py†L68-L129】
   - Capture shared numeric defaults (charge ramp multipliers, splash falloff) in `core/rules.py` to keep balancing centralized.

3. **Validation and tooling**
   - Add a JSON schema file (e.g., `data/troop_mechanics.schema.json`) plus a simple validator script in `tools/validate_troop_data.py` that runs in CI.
   - Update documentation comments inside the JSON to explain how simulator-only fields map to Royale API concepts.

## 3. Layer Mechanics Incrementally in the Engine

Deliver features in thin slices so every merge keeps the simulator runnable:

1. **Charge / dash framework**
   - Implement `Unit.start_charge()` and `Unit.resolve_charge_hit()` helpers to toggle the existing `_charge_progress`, `_charge_ready`, and `_charge_active` flags. 【F:core/simulation.py†L68-L91】
   - Track uninterrupted travel distance in `Engine._update_unit_motion` and trigger the windup threshold from mechanic data. When the target changes, reset the charge according to `reset_on_hit` or `reset_on_target_change` flags.
   - Add per-archetype logic: dashers (Bandit) use short windups and instant teleport arcs, chargers (Prince, Battle Ram) ramp movement speed, leapers (Mega Knight) queue vertical splash jumps.

2. **Spawn and death effects**
   - Introduce `Engine.queue_spawn_effect(unit, payload)` that reuses the `SpellEffect` interface to schedule timed pulses. 【F:core/simulation.py†L121-L170】
   - Extend `deal_area` to accept optional crowd-control payloads, so spawn zaps apply stun/slow alongside damage. Ensure stacking rules (e.g., max stun length) are enforced globally.
   - Allow death-trigger payloads to spawn additional units or spells via a shared helper so cards like Night Witch can reuse the machinery.

3. **Splash, chain, and piercing attacks**
   - Modify projectile resolution so splash radius pulls from `unit.splash_radius` and the new mechanic payload. 【F:core/simulation.py†L62-L91】
   - Add chain targeting by letting a projectile consume `chain_config` (max targets, chain radius, damage falloff). Iterate through candidates sorted by proximity and apply scaling modifiers. 【F:core/simulation.py†L173-L200】
   - Support ground-only splash vs. full 3D splash so units like Bowler (ground-only) and Executioner (air/ground) can coexist.

4. **Companion and attachment systems**
   - Use `support_units` to spawn child units at deploy time (Goblin Giant backpack) and `death_spawn_config` for on-death minions (Lava Pups). 【F:core/simulation.py†L74-L95】
   - Track parent/child relationships via `_support_children` so when a parent dies or is displaced, attachments detach or expire gracefully.

5. **Iterative polish**
   - After each mechanic lands, re-run the regression anchors, update the GUI overlays, and solicit balance feedback before proceeding to the next archetype.

## 4. Testing, Telemetry, and Tooling

1. **Automated tests**
   - Add pytest scenarios in `tests/` for each mechanic: charge timing, splash coverage, spawn-effect timing, companion spawn counts, crowd-control stacking rules. Use deterministic seeds to keep assertions reliable.
   - Provide golden JSON outputs describing unit timelines so reinforcement-learning consumers can diff behaviors automatically.

2. **Developer tooling**
   - Build CLI probes (`tools/sim_charge_test.py`, `tools/sim_spawn_effect_test.py`) that run headless fights and emit structured logs, making mechanic tuning scriptable.
   - Instrument the GUI with optional overlays: draw charge progress bars, splash radii, upcoming spawn-effect timers, and companion attachment links. This shortens the debug loop during balance passes.

3. **Telemetry hooks**
   - Emit structured events from the engine (e.g., `engine.emit_event({"type": "charge_hit", ...})`) so downstream analytics or RL pipelines can subscribe without parsing raw logs.
   - Add counters for mechanic usage and success rates to spot anomalies (e.g., charge never activating after a refactor).

## 5. Documentation and Release Hygiene

1. **Living knowledge base**
   - After every mechanic milestone, update `docs/project_knowledge.md` to reflect the new simulator state so contributors have an accurate roadmap. 【F:docs/project_knowledge.md†L1-L70】
   - Keep a `docs/troop_mechanics_changelog.md` that records behavior changes card-by-card, which is crucial for ML experiments depending on stable simulations.

2. **Contributor guidance**
   - Expand the README with a “Troop Mechanics Quickstart” section linking to this blueprint, the data schema, and sample tests. Include setup instructions for the CLI harnesses.
   - Document coding patterns (e.g., avoid mechanic-specific booleans in the engine; prefer data-driven payloads) to ensure future contributors extend the same systems instead of creating parallel code paths.

3. **Release management**
   - Bundle mechanic updates into labeled releases (e.g., `v0.4.0-chargers`) and publish release notes summarizing new behaviors, regressions fixed, and outstanding gaps.
   - Tag data-schema bumps so downstream consumers know when to re-export or migrate datasets.

## 6. Execution Roadmap

| Milestone | Scope | Deliverables | Owners | Target |
| --- | --- | --- | --- | --- |
| M0 | Baseline audit | Mechanics inventory table, regression scripts scaffold | Gameplay + Data team | Week 1 |
| M1 | Data schema v1 | Updated JSON, schema validator, docs | Data team | Week 2 |
| M2 | Chargers & dashers | Charge framework, Prince/Battle Ram/Bandit behaviors, tests | Gameplay | Week 4 |
| M3 | Spawn & splash | Mega Knight jump, Electro Wizard zap, splash AoE refactor, GUI overlays | Gameplay + UX | Week 6 |
| M4 | Companions & death spawns | Goblin Giant backpack, Lava Hound pups, Night Witch bats, tests | Gameplay | Week 7 |
| M5 | Polish & telemetry | RL telemetry hooks, changelog, release packaging | Gameplay + Platform | Week 8 |

Track progress through GitHub Projects, enforce feature flags where necessary, and refuse merges that regress the regression anchors. Following this blueprint keeps implementation disciplined while driving toward full troop-mechanic fidelity.
