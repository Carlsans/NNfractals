# 48-Hour Evolution Loop — Runbook

This is the procedure each scheduled wake-up follows. The evolution daemon
(`nnfractals`) runs independently via `scripts/evo_daemon.sh` (setsid-detached),
so it keeps evolving and auto-deduping even while the agent is rate-limited or
absent. The agent's job is periodic (every 6h) data-driven tuning.

State lives in `loop_state.json`:
`start_time`, `total_hours` (48), `interval_hours` (6), `loops_completed`,
`last_analysis_time`, `changes_log[]`.

## Every wake-up

1. `cd` to the project. Run `bash scripts/evo_daemon.sh ensure` — restarts the
   daemon if it died (crash, reboot, OOM). This is the resilience guarantee.
2. Read `loop_state.json`. Compute `now - last_analysis_time` and `now - start_time`.
3. **Stop condition:** if `now - start_time >= total_hours*3600` OR
   `loops_completed >= total_hours/interval_hours` (=8): do a final analysis,
   commit, write a closing note to `changes_log`, and DO NOT reschedule. Leave
   the daemon running (user can stop it with `evo_daemon.sh stop`).
4. **Analysis due** (`now - last_analysis_time >= interval_hours*3600`):
   a. `python3 scripts/loop_analyze.py --since <last_analysis_time>` — read it.
   b. Interpret: Are laion/clip rising vs the previous loop's note? Is
      self_replication rising? Which bases dominate the top quartile? Is the
      pool diverse or collapsing onto one formula family?
   c. Make 1–3 *targeted, justified* changes toward more beauty AND
      self-replication. Levers: `config.toml` (thresholds, mutation_rate/scale,
      novelty_weight, self_replication_weight, eval_max_iter), the basis bias in
      `Genome::random_exotic` (src/genome.rs), the beauty weighting
      (src/fitness.rs), or the self-replication metric itself. One change at a
      time is easier to attribute next loop.
   d. `cargo build --release` (must succeed). Then `bash scripts/evo_daemon.sh restart`.
   e. Update `loop_state.json`: `loops_completed += 1`, `last_analysis_time = now`,
      append a `changes_log` entry stating what changed and WHY.
   f. `git commit` the changes (config/code + loop_state.json).
   g. Reschedule (see below).
5. **Not yet due:** reschedule and stop (cheap wake — no analysis, few tokens).

## Rescheduling

Use ScheduleWakeup with `delaySeconds: 3600` (the max), `prompt` = the same
continuation instruction, `reason` = short status. Hourly wakes keep the daemon
healthy and land near each 6h boundary. Non-analysis wakes must stay minimal.

## Guardrails

- Never let a bad change persist: if a loop's change lowered scores vs the prior
  note, revert it as one of the next loop's changes.
- Keep the daemon as the source of truth — don't run a second `nnfractals`.
- All scores: laion [0,10], clip [0,1], beauty [0,1], self_replication [0,1].
- If `cargo build` fails after an edit, fix it before restarting; the daemon
  keeps running the old binary until a successful `restart`.
