# Marathon Prompt (mission order — CLAUDE.md is the constitution)

```
/research-pipeline "Execute the GRPO-Guard engineering project. READ CLAUDE.md in the
project root FIRST (constitution: clean repo, fixed workload Qwen2.5-1.5B + Countdown +
TRL/vLLM server, F1-F4 canonical faults, producer ownership + event lineage + content
hashing, guarded single-use update handle, reason-coded validation, paired replay,
Day3/4/5 gates, 40 GPU·h hard cap, forbidden legacy narratives). AUTHORITATIVE DESIGN:
GRPO-Guard_详细项目设计与旧项目迁移手册.md (project root) — read it fully before any
implementation; it locks schema/protocols/faults/tests/budget/release. ARIS role is
execution discipline, NOT re-discovery: no Phase 0-1 redesign; run the Day1-Day5 plan
with per-day review; before each of the three gates run an independent cross-model
review (contract-vs-implementation, fail-closed behavior, test completeness, claim
honesty — reviewer fallback per shared-references/reviewer-fallback.md). Decision log
for every deviation BEFORE first formal run (never after seeing results). Honesty:
no post-hoc gate loosening, frozen fixtures never auto-overwritten, published numbers
traceable to artifact+commit+checksum. Server TBD (reserved — candidates jindun/
autodl1/local; GPU contention rules of the chosen box). Budget 30-36 GPU·h hard cap 40,
resume-from-checkpoint default. Outputs: run_manifest.json + REPORT.md + SHA256SUMS per
run (no-overwrite), release package at Day 5 (README/demo/report/checksums/LICENSE),
resume bullets only from gate-passed numbers. Human owns the project. Start with
compatibility-profile freeze + official TRL+vLLM server smoke." --deep_mode: false,
auto_write: true, auto_proceed: true, venue: "engineering release (not a paper)"
```
