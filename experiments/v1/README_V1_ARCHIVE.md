# V1 pilot archive

V1 is the completed 200-question pilot and must remain scientifically and
operationally separate from V2.

Archive recorded: 2026-09-04

Git commit containing the versioned V1 code and paper outputs:

```text
3c223e5ffea3b872923c92bf422d7d8723d0fb80
```

## Canonical V1 artifacts

- Experiment configuration: `config/experiment.yaml`
- Model configuration: `config/models.yaml`
- Preregistration: `EXPERIMENT.md`
- Sample: `data/samples/mmlu_pro_pilot.jsonl`
- Sample manifest: `data/manifests/mmlu_pro_pilot_manifest.json`
- Raw checkpoint: `results/raw/pilot.sqlite3`
- Analysis manifest: `paper_outputs/analysis_manifest.json`
- Paper outputs: `paper_outputs/`

The raw checkpoint remains in its original location so existing V1 commands
continue to work. It is intentionally excluded from git because `results/` is
generated data. Back it up separately before deleting or replacing the local
workspace.

## SHA-256 checksums

```text
3c6365ba1450e639148477d879021c332574a81c469cab2a03348b7aef742e0f  results/raw/pilot.sqlite3
ab3ee47c415bc1343fe180f42985b021ed008a86431e34d16a4bb34a40712ef2  config/experiment.yaml
0d09c5bef90f0465bf85f472a78f308af91dc531584ee1b839c1bb67c6cb0f00  config/models.yaml
2f7e03a5e6032dd1c0ac6ba05cc06dd394e56f8e44ab231efedfe00024b6d104  data/samples/mmlu_pro_pilot.jsonl
ac0621e587b93e67d0366d32817768a575b8e2c965a1db666a4de23c729965af  data/manifests/mmlu_pro_pilot_manifest.json
a74357dc46f665c1e3e46ff118b49f7dbf26857d1a81790862506a6079670491  paper_outputs/analysis_manifest.json
```

## V2 reuse boundary

V1 Stage 1 and Stage 2 records may be reused only when the example, requested
model ID, model configuration, frozen answer dependency, and prompt version
match exactly.

V1 Stage 3 records must never be imported as V2 decisions. V1 used a
human-facing `RELY` / `VERIFY` contract; V2 uses the matched owner factorial
and `USE_UNVERIFIED` / `VERIFY_FIRST`.
