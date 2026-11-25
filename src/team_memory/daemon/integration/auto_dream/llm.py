"""LLM configuration resolution and JSON call helpers for AutoDreamWorker."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from team_memory.daemon.integration.auto_dream.state import _estimate_tokens
from team_memory.services.llm.llm_client import LLMClient, LLMError

logger = logging.getLogger("daemon.auto_dream")


class AutoDreamLLMMixin:
    """Mixin providing LLM settings resolution and fallback JSON chat calls."""

    def _resolve_llm_settings(
        self,
        model_override: str | None = None,
    ) -> tuple[str, str, str, str, str, float]:
        """Resolve (model, fallback_model, base_url, api_key, provider, timeout).

        Priority for model:
          1. CLI/call model_override
          2. TM_AUTO_DREAM_MODEL env var
          3. auto_dream.llm_model in config.yaml
          4. refinement.model in config.yaml
          5. ClientConfig.llm.model
          6. default ("gbu-gemini-3.8-flash")
        """
        client_llm = None
        try:
            from team_memory.client_config import ClientConfig

            client_llm = ClientConfig.load().llm
        except Exception:
            client_llm = None

        ref = self._refinement_cfg
        model = (
            (model_override or "").strip()
            or os.environ.get("TM_AUTO_DREAM_MODEL", "").strip()
            or (self._cfg.llm_model or "").strip()
            or (ref.model or "").strip()
            or (getattr(client_llm, "model", "") if client_llm else "").strip()
            or "gbu-gemini-3.8-flash"
        )
        fallback = (
            os.environ.get("TM_AUTO_DREAM_FALLBACK_MODEL", "").strip()
            or (self._cfg.llm_fallback_model or "").strip()
            or (ref.fallback_model or "").strip()
        )
        if fallback == model:
            fallback = ""

        base_url = (
            os.environ.get("TM_LLM_BASE_URL", "").strip()
            or (ref.base_url or "").strip()
            or (getattr(client_llm, "base_url", "") if client_llm else "").strip()
            or "https://gw.gburouter.com/v1"
        )
        api_key = ""
        if ref.api_key_env:
            api_key = os.environ.get(ref.api_key_env, "").strip()
        if not api_key and client_llm is not None:
            api_key = (client_llm.resolved_api_key or "").strip()

        provider = (ref.provider or "litellm").strip()
        timeout = float(self._cfg.llm_timeout or ref.timeout or 60)
        return model, fallback, base_url, api_key, provider, timeout

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model_override: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Call LLM expecting JSON output, with fallback_model retry.

        Returns (parsed_dict, estimated_input_tokens, estimated_output_tokens).
        """
        in_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)

        if self._llm_client is not None:
            result = await self._llm_client.chat_json(
                system_prompt,
                user_prompt,
                temperature=0.2,
                timeout=float(self._cfg.llm_timeout or 60),
            )
            if not isinstance(result, dict):
                raise LLMError("LLM returned non-object JSON")
            out_tokens = _estimate_tokens(json.dumps(result, ensure_ascii=False))
            return result, in_tokens, out_tokens

        model, fallback, base_url, api_key, provider, timeout = self._resolve_llm_settings(model_override)
        models_to_try = [model] + ([fallback] if fallback else [])
        last_exc: Exception | None = None

        for idx, cur_model in enumerate(models_to_try):
            client = LLMClient(
                base_url=base_url,
                model=cur_model,
                api_key=api_key,
                provider=provider,
            )
            try:
                result = await client.chat_json(
                    system_prompt,
                    user_prompt,
                    temperature=0.2,
                    timeout=timeout,
                )
                if not isinstance(result, dict):
                    raise LLMError("LLM returned non-object JSON")
                out_tokens = _estimate_tokens(json.dumps(result, ensure_ascii=False))
                return result, in_tokens, out_tokens
            except Exception as exc:
                last_exc = exc
                if idx < len(models_to_try) - 1:
                    logger.warning(
                        "AutoDream LLM model %s failed (%s), falling back to %s",
                        cur_model,
                        exc,
                        models_to_try[idx + 1],
                    )

        raise last_exc or LLMError("AutoDream LLM call failed")
