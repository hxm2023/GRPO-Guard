# Security

## Trust model (design doc §5)

GRPO-Guard v0.1 detects configuration and integration errors between trusted
components (trainer control plane, runtime adapter, scorer adapter, event
store, validator).  It does NOT defend against a malicious producer that
forges events, tensors, checkpoint hashes and canary results simultaneously —
that would require cryptographic non-repudiation, which is explicitly out of
scope (design doc §5.3).  Reported language is therefore "validate, detect,
reject, quarantine, audit evidence" — never "absolute proof" or
"tamper-proof".

## Reporting

This is an engineering showcase project.  If you find a way to bypass the
guarded update (e.g., a path that lets a rejected envelope reach the
optimizer, or a nonce-reuse path), open an issue with a minimal repro.

## Fail-closed guarantees that ARE enforced

- The guarded update adapter accepts only a `ValidatedBatchHandle` whose
  referenced pre-update decision is `ALLOW`; absence of a decision verifier
  is a hard error.
- Nonce reuse raises before the optimizer; tokenizer re-calls are refused.
- Artifact content hashes are re-verified at update time.
- `quarantine`/`reject` decisions never enter reward or update consumption.
