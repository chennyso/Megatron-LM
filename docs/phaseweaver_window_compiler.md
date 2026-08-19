# PhaseWeaver Window Compiler

PhaseWeaver is a narrow optimization layer for Megatron's existing PP/VPP and
distributed-optimizer execution.  Its input is a phase trace containing PP
activation P2P and DP/ZeRO collective issue/completion events.  It does not
change tensor ownership, NCCL ticket order, or training semantics.

The compiler treats the next VPP period frontier as the scheduling contract.
For each movable DP bucket it records a release time, a service estimate, a
deadline before the next critical PP frontier, and a communicator ticket.  It
then emits a finite set of phase modes (`eager`, `dephase-*`).  A mode is
accepted only when it carries all four frontier certificates:

```text
activation_fifo_prefix
collective_ticket_prefix
vpp_cursor_unchanged
zero_outstanding_debt
```

The offline command is:

```bash
python tools/phaseweaver_synthesize.py \
  --trace-glob 'run/traces/**/*.phase.jsonl' \
  --output phaseweaver_modes.json
```

The current compiler is deliberately a candidate generator and legality
checker.  It must not be described as a speedup until a ticketed runtime
executes its modes and held-out runs compare it with eager overlap,
PP-first/DP-first order, and an oracle schedule.
