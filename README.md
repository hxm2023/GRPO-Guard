# GRPO-Guard

Trajectory contract, lineage and fault-injection framework for online LLM
post-training (TRL GRPO + vLLM server). v0.1 builds a machine-verifiable
evidence chain between trainer / rollout / behavior scoring / reward /
optimizer update, detecting silent errors: static rollout policy, misbound
old-logprob, retokenization, and mask shift.

> Status: **WIP — Day 1 of the five-day plan**. No gate has passed yet; nothing
> here is a finished engineering claim. The authoritative design is
> `GRPO-Guard_详细项目设计与旧项目迁移手册.md` (project root, v1.0).

## Quickstart (not yet public — placeholder)

```bash
uv sync --frozen
uv run grpo-guard contract-check --cases tests/frozen/normal
```

Documentation, architecture diagram, demo and release artifacts land after
the Day 3/4/5 gates per the design doc §16/§23.

## License

Apache-2.0.
