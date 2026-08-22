"""F1-F4 canonical fault injectors (design doc §11, §12.3).

Every injector takes the canonical happy-path trajectory from
``grpo_guard.testing`` and mutates EXACTLY ONE target field, rebuilding the
affected sealed events.  These are minimal fault fixtures
``reconstructed_from_incident`` — they are NOT ports of the legacy trainer
code (design doc §3.2).
"""

from grpo_guard.faults.mask_shift import inject_f4_mask_shift
from grpo_guard.faults.misbound_logprob import inject_f2_misbound_logprob, inject_f2_wrong_generation
from grpo_guard.faults.retokenization import (
    inject_f3_retokenization,
    inject_f3_retokenized_sequence,
    inject_f3_template_variant,
)
from grpo_guard.faults.static_rollout import inject_f1_stale_sync, inject_f1_static_rollout

__all__ = [
    "inject_f1_stale_sync",
    "inject_f1_static_rollout",
    "inject_f2_misbound_logprob",
    "inject_f2_wrong_generation",
    "inject_f3_retokenization",
    "inject_f3_retokenized_sequence",
    "inject_f3_template_variant",
    "inject_f4_mask_shift",
]
