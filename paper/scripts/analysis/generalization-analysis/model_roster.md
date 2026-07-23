# MultiZebra model roster

Models for the MultiZebra transfer runs: what we already have, plus the proposed
additions. The additions target what the GSM control test flagged as missing — more
**families** (the interaction died with only 2), an **English-centric** family (to
restore the resource gradient that EuroLLM/Qwen mute), and a **second reasoning
family** (only Qwen has on/off pairs today). Where a family overlaps GSM, use the same
checkpoints so the cross-task comparison stays clean.

| Have | Model ID | Family | Size | Reasoning | Role |
|:---:|---|---|---|---|---|
| ✓ | `Qwen/Qwen3.5-4B` | Qwen3.5 | 4B | on **+** off | Broad-multilingual anchor; reasoning pair |
| ✓ | `Qwen/Qwen3.5-9B` | Qwen3.5 | 9B | on **+** off | " |
| ✓ | `Qwen/Qwen3.5-27B` | Qwen3.5 | 27B | on **+** off | " |
|  | `Qwen/Qwen3.5-0.8B` | Qwen3.5 | 0.8B | on **+** off | Small-size anchor |
| ✓ | `utter-project/EuroLLM-1.7B-Instruct` | EuroLLM | 1.7B | off | EU-multilingual anchor |
| ✓ | `utter-project/EuroLLM-9B-Instruct-2512` | EuroLLM | 9B | off | " |
| ✓ | `utter-project/EuroLLM-22B-Instruct-2512` | EuroLLM | 22B | off | " |
|  | `allenai/OLMo-3-7B-Think` | OLMo-3 | 7B | on | English-centric; 2nd reasoning family |
|  | `allenai/OLMo-3-7B-Instruct` | OLMo-3 | 7B | off | " |
|  | `allenai/OLMo-3-32B-Think` | OLMo-3 | 32B | on | " |
|  | `allenai/OLMo-3-32B-Instruct` | OLMo-3 | 32B | off | " |
|  | `allenai/OLMo-2-0425-1B-Instruct` | OLMo-2 | 1B | off | Small-size anchor (English-centric) |
|  | `google/gemma-3-1b-it` | Gemma-3 | 1B | off | Small-size anchor |
|  | `google/gemma-3-4b-it` | Gemma-3 | 4B | off | 3rd multilingual family; decouples size from family |
|  | `google/gemma-3-27b-it` | Gemma-3 | 27B | off | " |

**Totals.** Have: 9 runs (Qwen 3×2 on/off + EuroLLM 3). Add: ~10 runs (OLMo-3 2×2 on/off +
Gemma-3 1b/4b/27b + Qwen-0.8B on/off + OLMo-2-1B). → ~19 runs, sizes **0.8–32B**, four
families, two reasoning families.

**Notes.**
- Reasoning-off for Qwen is the same checkpoint with a `#no-thinking` flag, not a separate
  repo; OLMo-3 uses distinct `Think` / `Instruct` checkpoints.
- Confirm the exact OLMo-3 checkpoint IDs against the `OLMo 3` rows already in the GSM
  roster, so the runs match.
- **Raise the generation token cap** (currently 8k) on all reasoning-**on** runs (Qwen +
  OLMo) — at 8k, ~33% of reasoning-on predictions truncate to empty, which forces the
  exclude-vs-count-as-wrong sensitivity analysis. A higher cap dissolves it.
- **Span the size range down to ~1B — that, not family count or model number, is what
  identifies the `resource × size` interaction and resource's share of the transfer
  variance.** GSM test (Germanic-only): a 1.7B floor gives interaction +0.002 (ns) and
  variance ~0; adding ~1B models flips it to −0.031 (p=0.03) and variance 10%, matching
  full-GSM's sign. The tiny-vs-large contrast is what anchors "big models depend less on
  resource"; adding mid/large families alone (Gemma/OLMo at 4B+) does *not* recover it.
- Extra families still earn their place — English-centric resource range, a 2nd reasoning
  family, tokenizer diversity for fertility, family-level robustness — but for the
  interaction/variance specifically, prioritize the **size span** over adding families.
  Small models are also the cheapest to run, so the size-span fix is nearly free.
