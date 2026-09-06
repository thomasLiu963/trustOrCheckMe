# V2 protocol packet index

These are the source files that define the frozen V2 experiment.
Stage 3 has no four separate prompt files. One builder in
`src/v2_prompts.py` assembles all owner × visibility × family variants.

## Requested items

| Requested item | File | Where |
|---|---|---|
| Stage 1 prompt | `src/prompts.py` | `STAGE_1_TEMPLATE`, `build_stage_1_prompt` |
| Stage 2 confidence prompt | `src/prompts.py` | `STAGE_2_TEMPLATE`, `build_stage_2_prompt` |
| Stage 3 human-owner hidden | `src/v2_prompts.py` | `build_verification_prompt` with `decision_owner=human`, `confidence_visibility=hidden`, `prompt_family=v2_owner_match_v1` |
| Stage 3 human-owner visible | `src/v2_prompts.py` | same, `confidence_visibility=visible` |
| Stage 3 AI-owner hidden | `src/v2_prompts.py` | `decision_owner=ai_system`, `confidence_visibility=hidden` |
| Stage 3 AI-owner visible | `src/v2_prompts.py` | `decision_owner=ai_system`, `confidence_visibility=visible` |
| Robustness paraphrase prompt | `src/v2_prompts.py` | same builder with `prompt_family=v2_owner_match_paraphrase_v1` |
| Exact output constraints / parsing | `src/schemas.py`, `src/prompts.py`, `src/v2_prompts.py`, `src/model_adapters.py` | Pydantic payloads; Stage 1/2 parsers in `prompts.py`; Stage 3 parser `parse_verification_response`; provider JSON schemas in `output_schema` |
| Model / API IDs and generation parameters | `config/models.yaml`, `src/model_adapters.py` | per-provider request construction |
| Google 512-token amendment | `EXPERIMENT_V2.md`, `config/models.yaml` | protocol text; `google_gemini38_flash.max_output_tokens: 512` |
| MMLU-Pro question selection | `config/experiment_v2.yaml`, `src/v2_datasets.py`, `src/datasets.py` | config sizes/revision; V2 sample construction; `select_category_stratified` |
| Selected question IDs | `data/manifests/` | `mmlu_pro_v2a_manifest.json`, `mmlu_pro_v2b_manifest.json`, `mmlu_pro_robustness_manifest.json` |
| Bootstrap / statistics | `src/bootstrap.py`, `src/v2_analysis.py`, `src/v2_scoring.py`, `src/calibration.py` | 5,000-resample question-level bootstrap; scoring and isotonic calibration |

## Notes

- V1 Stage 3 (`RELY` / `VERIFY`) is in `src/prompts.py` as `STAGE_3_TEMPLATE`. V2 Stage 3 (`USE_UNVERIFIED` / `VERIFY_FIRST`) is only in `src/v2_prompts.py`.
- Visible Stage 3 adds one sentence containing the frozen Stage 2 probability. Hidden Stage 3 omits that sentence and does not re-elicit confidence.
- Hidden vs visible, and human vs AI, are not stored as four templates. Owner wording is `_owner_language`; visibility is `confidence_sentence` inside `build_verification_prompt`.
- Protocol narrative and hypotheses: `EXPERIMENT_V2.md`. Runtime config: `config/experiment_v2.yaml`.
