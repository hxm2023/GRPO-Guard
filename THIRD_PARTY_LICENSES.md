# Third-party licenses

GRPO-Guard is Apache-2.0.  All runtime/test dependencies use permissive
licenses (design doc §19.1 license audit):

| package | license |
|---|---|
| pydantic | MIT |
| numpy | BSD-3-Clause |
| pyyaml | MIT |
| hypothesis | MPL-2.0 |
| pytest | MIT |
| colorama (via pytest) | BSD-3-Clause |
| jinja2 (via pytest) | BSD-3-Clause |
| torch (test/gpu extra) | BSD-3-Clause |
| transformers (gpu extra) | Apache-2.0 |
| trl (gpu extra) | Apache-2.0 |
| vllm (gpu extra) | Apache-2.0 |
| accelerate (gpu extra) | Apache-2.0 |
| peft (gpu extra) | Apache-2.0 |
| datasets (gpu extra) | Apache-2.0 |
| safetensors (gpu extra) | Apache-2.0 |

Model: Qwen/Qwen3-4B (Apache-2.0), revision
1cfa9a7208912126459214e8b04321603b3df60c.

Audited 2026-08-23.
