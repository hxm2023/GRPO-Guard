# GRPO-Guard Day 2 — guarded online closed loop

- run: loop-1787355683
- v0 rollout sequences: 32
- identity ALLOW: 32
- pre-update ALLOW: 32
- committed optimizer steps: 1
- upstream sync params observed: 398
- canary: {'calibration_reloads': 5, 'tolerance': 0, 'v1_verdict': 'pass', 'v1_drift': {'max_token_drift': 0}}
- v1 rollout sequences: 2
- update metrics: {'ratio_p50': 1.0, 'ratio_p95': 1.083100438117981, 'ratio_max': 2.4342923164367676, 'clip_fraction': 0.02197265625, 'loss': 0.0, 'B': 32, 'T': 90, 'group_size': 4}
