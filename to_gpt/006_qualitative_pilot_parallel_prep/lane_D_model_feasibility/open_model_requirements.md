# Open-model requirements (planning only)

No open-weight inference hook exists in `src/model_adapters.py`. `create_adapter` supports openai, anthropic, google, and xai only.

Scanner hits in the adapter file (should be false): `{'vllm': False, 'ollama': False, 'huggingface': False, 'transformers': False, 'llama.cpp': False, 'local_model': False}`

This lane did **not** install models or download weights.

## What would be required later

1. **One instruction-tuned open model**
   - New provider in `create_adapter` (vLLM, Hugging Face, or similar)
   - Frozen `api_model` string and tokenizer
   - Same Stage-1/2/3 JSON schema (`USE_UNVERIFIED` / `VERIFY_FIRST`, probability in [0,1])
   - Deterministic request keys and CheckpointStore compatibility
   - Cost/runtime accounting even if local GPU is "free"

2. **Log probability access**
   - Not available from the current GPT/Claude/Gemini/Grok adapters in this repo
   - An open model would need an explicit logprob API (token-level `logprob` of the frozen answer, or sequence logprob)
   - That is a new scientific construct, not a drop-in replacement for verbal q1

3. **Hidden activation access**
   - Not implemented. Would require a white-box runtime (hooks on residual streams), a stored activation policy, and a protocol addendum
   - Task 006 forbids jumping to interpretability

4. **Same frozen-answer / confidence / verification structure**
   - Reuse `src/prompts.py` and `src/task006_prompts.py`
   - Do not mix numeric Study-1 L/C prompts with qualitative families in one unfrozen comparison

Status: **BLOCKED** for open-weight execution in the current codebase (missing adapter). Engineering, not a scientific stop on GPT/Claude work.
