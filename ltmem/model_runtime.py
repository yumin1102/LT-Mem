import importlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv
from typing import Optional, Any, Dict, Tuple, cast


@dataclass
class LocalModelConfig:
    context_window: int = 8192
    use_4bit_env_default: bool = True


@dataclass
class ModelProvider:
    kind: str
    client: Any = None
    model_name: Optional[str] = None
    llm: Optional[Any] = None


class ModelRuntime:
    def __init__(
        self, hf_config: Optional[LocalModelConfig] = None, *, provider_kind: Optional[str] = None
    ) -> None:
        self.hf_config = hf_config or LocalModelConfig()
        self.provider_kind = provider_kind
        self.provider: Optional[ModelProvider] = None

    def _ensure_provider(self) -> ModelProvider:
        if self.provider is None:
            self.provider = init_model_provider(
                hf_config=self.hf_config,
                kind=self.provider_kind,
            )
        return self.provider

    def generate(self, prompt: str) -> str:
        provider = self._ensure_provider()
        if provider.kind == "qwen":
            return cast(LocalTransformersLLM, provider.llm).generate(prompt)
        response = provider.client.models.generate_content(
            model=cast(str, provider.model_name),
            contents=prompt,
            config=create_generate_content_config(temperature=0.0, max_output_tokens=4096),
        )
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError(f"{provider.kind} returned no text")
        return response.text.strip()

    def select_choice(self, prompt: str, choices: Tuple[str, ...]) -> str:
        provider = self._ensure_provider()
        if provider.kind == "qwen":
            local_llm = cast(LocalTransformersLLM, provider.llm)
            choice = local_llm.select_choice(prompt, choices)
        else:
            response = provider.client.models.generate_content(
                model=cast(str, provider.model_name),
                contents=prompt,
                config=create_generate_content_config(
                    temperature=0.0,
                    candidate_count=1,
                    response_mime_type="application/json",
                    response_json_schema={"type": "string", "enum": list(choices)},
                ),
            )
            text = response.text
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{provider.kind} returned no text")
            choice = json.loads(text.strip())
        if not isinstance(choice, str) or choice not in choices:
            raise ValueError(f"provider returned unsupported choice: {choice!r}")
        return choice


class LocalTransformersLLM:
    def __init__(
        self,
        *,
        model_path: str,
        context_window: int,
        model_kwargs: Dict[str, Any],
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_safetensors=True,
            **model_kwargs,
        )
        self.model.eval()
        self.context_window = context_window

    def _format_chat_prompt(self, prompt: str) -> str:
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(self._format_chat_prompt(prompt), return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[-1]
        max_new_tokens = 512
        context_limit = min(32768, self.model.config.max_position_embeddings)
        if prompt_length + max_new_tokens > context_limit:
            raise ValueError("QA memory exceeds the model context window; use a smaller snapshot")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()

    def select_choice(self, prompt: str, choices: Tuple[str, ...]) -> str:
        import torch

        encoded_choices = [
            (choice, self.tokenizer.encode(choice, add_special_tokens=False)) for choice in choices
        ]
        max_choice_tokens = max(len(token_ids) for _choice, token_ids in encoded_choices)
        formatted_prompt = self._format_chat_prompt(prompt)
        encoded_prompt = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, self.context_window - max_choice_tokens),
        )
        encoded_prompt = encoded_prompt.to(self.model.device)
        prompt_ids = encoded_prompt["input_ids"]
        prompt_mask = encoded_prompt.get("attention_mask")
        prompt_length = prompt_ids.shape[-1]
        scored = []
        with torch.inference_mode():
            for choice, token_ids in encoded_choices:
                choice_ids = torch.tensor(
                    [token_ids], device=prompt_ids.device, dtype=prompt_ids.dtype
                )
                input_ids = torch.cat((prompt_ids, choice_ids), dim=-1)
                model_inputs = {"input_ids": input_ids}
                if prompt_mask is not None:
                    choice_mask = torch.ones(
                        (prompt_mask.shape[0], len(token_ids)),
                        device=prompt_mask.device,
                        dtype=prompt_mask.dtype,
                    )
                    model_inputs["attention_mask"] = torch.cat((prompt_mask, choice_mask), dim=-1)
                logits = self.model(**model_inputs).logits
                continuation_logits = logits[
                    :, prompt_length - 1 : prompt_length + len(token_ids) - 1, :
                ]
                log_probs = torch.log_softmax(continuation_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
                scored.append((float(token_log_probs.mean().item()), choice))
        return max(scored, key=lambda item: item[0])[1]


def _require_pipeline_cloud_egress(provider: str) -> None:
    approved = os.getenv("LT_MEM_ALLOW_CLOUD_PIPELINE_EGRESS", "").strip().casefold()
    if approved != "true":
        raise RuntimeError(
            f"{provider} requires explicit pipeline egress approval: "
            "set LT_MEM_ALLOW_CLOUD_PIPELINE_EGRESS=true after data-export review"
        )


def init_model_provider(
    *, hf_config: Optional[LocalModelConfig] = None, kind: Optional[str] = None
) -> ModelProvider:
    kind = (kind or os.getenv("LLM_PROVIDER", "qwen")).strip().lower()
    if kind not in {"gemini", "qwen"}:
        raise RuntimeError("LLM_PROVIDER must be gemini or qwen")
    if kind == "gemini":
        _require_pipeline_cloud_egress(kind)
        model_name = os.getenv("GEMINI_MODEL") or os.getenv("VERTEX_MODEL", "gemini-3.8-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            client = create_google_client(vertexai=False, api_key=api_key)
        else:
            client = create_google_client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        return ModelProvider(kind=kind, client=client, model_name=model_name)

    hf_config = hf_config or LocalModelConfig()
    from ltmem.local_model_integrity import (
        assert_artifact_unchanged,
        capture_artifact_fingerprint,
        enforce_read_only_model_policy,
        verify_local_model,
    )

    if os.getenv("QWEN_REVISION", "").strip():
        raise RuntimeError(
            "QWEN_REVISION is not supported for local Qwen; use QWEN_MODEL_PATH "
            "and QWEN_MODEL_SHA256"
        )
    model_path, model_sha256 = verify_local_model(
        os.getenv("QWEN_MODEL_PATH", ""),
        os.getenv("QWEN_MODEL_SHA256", ""),
        os.getenv("QWEN_MODEL_MANIFEST") or None,
        model_label="Qwen model",
        path_env="QWEN_MODEL_PATH",
        sha_env="QWEN_MODEL_SHA256",
    )
    enforce_read_only_model_policy(
        model_path,
        specific_env="QWEN_REQUIRE_READ_ONLY_MODEL",
    )
    model_fingerprint = capture_artifact_fingerprint(model_path)
    hf_model_name = str(model_path)

    from transformers import BitsAndBytesConfig
    import torch

    use_4bit = os.getenv("USE_4BIT", str(hf_config.use_4bit_env_default).lower()).lower() == "true"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": True,
        "device_map": "auto",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        model_kwargs["dtype"] = torch.float16

    local_llm = LocalTransformersLLM(
        model_path=hf_model_name,
        context_window=hf_config.context_window,
        model_kwargs=model_kwargs,
    )
    assert_artifact_unchanged(model_path, model_fingerprint, artifact_label="Qwen model")
    print(f"[LLM Provider] Qwen initialized: {hf_model_name} [sha256: {model_sha256[:12]}...]")
    return ModelProvider(kind=kind, llm=local_llm, model_name=hf_model_name)


_INSTALL_HINT = (
    "Google Gemini/Vertex provider requires google-genai; "
    "rebuild the Docker image with python3 scripts/build_image.py"
)


@lru_cache(maxsize=1)
def _load_google_genai() -> Tuple[Any, Any]:
    try:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return genai, types


def create_google_client(**kwargs: Any) -> Any:
    genai, _types = _load_google_genai()
    return genai.Client(**kwargs)


def create_generate_content_config(**kwargs: Any) -> Any:
    _genai, types = _load_google_genai()
    return types.GenerateContentConfig(**kwargs)


load_dotenv()
