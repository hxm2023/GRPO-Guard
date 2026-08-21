"""Policy and split manifests (design doc §7.2, §6.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from grpo_guard.schema.artifacts import ArtifactRef


class PolicyManifest(BaseModel):
    manifest_id: str
    model_id: str
    model_revision: str
    policy_version: int = Field(ge=0)
    parent_policy_version: int | None = Field(default=None, ge=0)
    weights: list[ArtifactRef] = Field(default_factory=list)
    checkpoint_manifest_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    precision: str
    adapter_kind: Literal["full", "lora", "qlora"]
    base_model_sha256: str | None = None
    adapter_sha256: str | None = None
    code_commit_sha: str
    config_sha256: str


class SplitManifest(BaseModel):
    split_id: str
    split_name: Literal["train", "calibration", "held_out"]
    prompt_ids: list[str] = Field(default_factory=list)
    content_sha256s: dict[str, str] = Field(default_factory=dict)
