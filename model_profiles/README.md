# model_profiles/

Hand-authored ground-truth records of what each OpenAI-family model snapshot
actually supports. These YAML files drive capability filtering: a third-party
model configured with `profile: gpt-5.4-mini` runs exactly the test subset
that OpenAI's `gpt-5.4-mini` passes. See
[docs/profile-based-compatibility-testing.md](../docs/profile-based-compatibility-testing.md)
for the full design.

## Adding a new profile

**Rule of thumb: never copy the PASS list blindly.** A passing test could mean
the model actually supports the feature *or* that the test's assertions are
too loose. Every capability entry must be a judgement call, not a line in a
report.

1. Point `config.yaml` at the official endpoint (`https://api.openai.com`,
   your real API key, `api_format: openai`, list only the model under study).
2. Run with the recording switch so marker-based filtering is bypassed:

   ```bash
   uv run pytest --config --ignore-profile tests/openai_compat/ -v
   ```

3. Read `reports/{timestamp}/summary.md`. For each test, classify the outcome:

   | Result | Action |
   |--------|--------|
   | PASS and the capability is real | add the marker to `capabilities:` |
   | FAIL — test/fixture bug | **do not** add; fix the test first |
   | FAIL — model truly rejects the parameter | leave out; note in comment |
   | Ambiguous / needs debug | shelf it, don't guess |

4. Create `model_profiles/openai/{model}/{YYYY-MM-DD}.yaml`. Use a *dated
   snapshot* as `snapshot:` (e.g. `gpt-4o-2024-08-06`) — the moving alias goes
   in `model:` for discovery only. Mirror the schema of an existing profile.
5. Human review before commit.

## Picking a snapshot

If `profile_snapshot:` is omitted in a config, the registry picks the file
with the latest `created_at`. Pin explicitly when you want a run to be
reproducible across future profile revisions.
