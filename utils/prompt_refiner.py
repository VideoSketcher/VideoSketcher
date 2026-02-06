import logging
import os
import threading
from typing import Dict, List, Optional

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_EXAMPLES = """
Concept: butterfly
Output: Step by step sketch process of a butterfly, following this drawing order: 1. Body – long vertical oval. 2. Right wing – large rounded shape attached to body. 3. Left wing – mirror of right wing. 4. Head – small circle atop body. 5. Antennae – two curved lines from head, ending in curls.
Concept: car
Output: Step by step sketch process of a car, following this drawing order: 1. Wheels – two circles spaced apart. 2. Body – rounded rectangle connecting wheels. 3. Windows – smaller rectangles on top. 4. Headlight – small circle at front. 5. Door – vertical line on body.
Concept: tree
Output: Step by step sketch process of a tree, following this drawing order: 1. Trunk – two vertical lines. 2. Canopy – large cloud shape from top of trunk. 3. Branches – smaller lines visible within canopy. 4. Roots – short angled lines at trunk base.
"""

_SYSTEM_PROMPT = f"""
You are a drawing instruction generator. Convert any given concept into clear, step-by-step sketch instructions.

**Output Format:**
"Step by step sketch process of a [concept], following this drawing order: 1. [Part name] – [simple shape description and position]. 2. [Part name] – [simple shape description and position]. ..."
**Ordering Principles (in priority):**
1. **Iconic features first** – Start with the parts that make this thing instantly recognizable (cat → ears + face shape; elephant → trunk; Eiffel Tower → tapered lattice frame)
2. **Anchoring structure** – Include the core element that other parts attach to, but only if needed for positioning
3. **Distinctive before generic** – Unique characteristics before common shapes
4. **Connected parts in sequence** – Parts that attach to each other should be drawn in logical connection order

Rules:
- Output exactly ONE line. Do not include any newline characters.
- Use this exact template: Step by step sketch process of a [concept], following this drawing order: 1. 2. 3. ..."
- Describe each part with a simple shape (oval, circle, rectangle, curved line, etc.).
- Include position relative to other parts: "above," "attached to," "extending from"
- Keep descriptions concise (under 15 words per step).
- Provide 5-10 steps for most subjects.
- Keep the language consistent with the user input.
- Do not add numbering or bullet points outside the required format.

Examples:
{_EXAMPLES}
"""


def _to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


class PromptRefiner:
    """Refine a concept into step-by-step sketch instructions via LLM.

    Supports two backends:
      - "openai": calls OpenAI-compatible chat API.
      - "qwen": runs a local Qwen model.
    """

    def __init__(
        self,
        backend: str = "openai",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
    ):
        assert backend in ("openai", "qwen")
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        # Lazy-loaded Qwen model
        self._qwen_model: Optional[torch.nn.Module] = None
        self._qwen_tokenizer = None
        self._qwen_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(self, raw_prompt: str) -> str:
        messages = self._build_messages(raw_prompt)
        if self.backend == "qwen":
            return self._refine_qwen(messages)
        return self._refine_openai(messages)

    def warmup(self) -> None:
        """Pre-load the Qwen model (no-op for openai backend)."""
        if self.backend == "qwen":
            self._load_qwen()

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(raw_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert this concept into drawing instructions: {raw_prompt}\n"},
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        assert lines, "empty response from LLM"
        return lines[0]

    # ------------------------------------------------------------------
    # Qwen local backend
    # ------------------------------------------------------------------

    def _resolve_dtype(self):
        if self.torch_dtype == "auto":
            return "auto"
        return _to_torch_dtype(self.torch_dtype)

    def _load_qwen(self):
        if self._qwen_model is not None:
            return self._qwen_tokenizer, self._qwen_model

        with self._qwen_lock:
            if self._qwen_model is not None:
                return self._qwen_tokenizer, self._qwen_model

            dtype = self._resolve_dtype()
            self._qwen_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "auto" else None,
            )
            if self.device != "auto":
                self._qwen_model = self._qwen_model.to(self.device)
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logging.info("Loaded Qwen local model: %s", self.model_name)
            return self._qwen_tokenizer, self._qwen_model

    def _refine_qwen(self, messages: List[Dict[str, str]]) -> str:
        tokenizer, model = self._load_qwen()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return self._normalize(content)

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _refine_openai(self, messages: List[Dict[str, str]]) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.environ.get("LLM_MODEL", "gpt-5.2")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content.strip()
        return self._normalize(content)

