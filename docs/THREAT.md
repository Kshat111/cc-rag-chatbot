# Threat model

How a malicious actor (or a plain mistake) could abuse the system. Status is honest: what is
built, what is only designed, what is a known gap. Boundaries refer to ARCHITECTURE.md.

| # | Threat | Boundary | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| T1 | Prompt injection through a document ("ignore your rules ...") | 2 | Model follows document text instead of the system rules | Passages sit in delimited blocks; system rule 5 says they are untrusted data; model has no tools, no secrets, text-only output (D-013) | Designed and covered by a prompt test. **Not tested against a live model** (no key in the build environment): run the planted-instruction test before the demo |
| T2 | Prompt injection by the user ("print your prompt", "answer from your own knowledge") | 1 | Reveals the system prompt (not secret: it is in the repo) or breaks grounding | Same prompt; no secret in the prompt; refusal sentence; invalid-citation check | Designed. Adversarial queries not yet run live |
| T3 | Answer outside the context (hallucination) | 5 | Confident wrong answer | Strict prompt, temperature 0, citations by id, refusal sentence, citation check (D-013) | Partial: citations are checked for existence, not for support. No retrieval-score threshold yet |
| T4 | Cost and abuse: huge or many queries | 1, 4 | API spend, local slowdown | `MAX_QUERY_CHARS` 500, history capped (5 turns / 4000 chars), answer capped at 500 tokens, queries over the embedder's 512-token limit are rejected | Partial: **no rate limit** in the UI yet |
| T5 | API key leakage | 4 | Account misuse | Key read from the environment by name only, sent as a header, never logged or printed; `.env` gitignored; keys are not in config.py | Done. Error messages include the provider's error text, never the request headers |
| T6 | Poisoned or changed corpus / model | 2, 3 | Wrong answers | Manifest sha256 and exact revision permalinks; index staleness check including a probe vector for a changed model (D-007); only data/raw is indexed | Done |
| T7 | Bad rewrite steers retrieval | 1, 4 | Wrong chunks retrieved | Invalid rewrites discarded; original question kept in the answer prompt; rewrite shown in the CLI (D-012) | Partial: a wrong-entity rewrite is not detected |
| T8 | Local LLM endpoint is attacker-controlled | 4 | Arbitrary text shown to the user | The base URL comes only from config/env set by the operator | Accepted: operator trusts their own endpoint |
