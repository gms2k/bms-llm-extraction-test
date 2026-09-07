# BMS LLM Rule Extraction Test

Part of COS40005 — Developing an AI Tool for Automated BMS Functional Compliance
Verification Using LLMs and Knowledge Graphs (team UG_COSEAT_02).

This tests local LLMs (via [Ollama](https://ollama.com)) on Stage 2 of the
pipeline: extracting a control rule from a Building Management System
Functional Description into strict JSON, instead of a free-text summary.

## What it does

`llama_extraction_test.py` takes two rules from the real FD (Rev. H) and asks
a model to extract each into a fixed schema (`rule_id`, `trigger_parameter`,
`controlled_parameter`, `condition_type`, `thresholds`/`branches`,
`applies_to`, `source_text`). The output is scored against a ground truth we
hand-verified from the PDF.

Two test rules, matching the examples Morshed used in the week 3 meeting:

1. **Easy** — CO2 damper control (§1.1.10), a proportional rule
2. **Tricky** — occupied-mode temperature setpoint (§1.1.13), a
   conditional-branching rule with two cases

Can run against hardcoded copy-pasted section text, or against the real FD
PDF directly (`--pdf`, using Docling to convert to markdown and slice out the
sections).

## Usage

```
ollama pull qwen2.5:7b        # or llama3.2:1b / qwen2.5:32b / qwen3:30b-a3b
pip install ollama
pip install docling           # only needed for --pdf

python llama_extraction_test.py --model qwen2.5:7b
python llama_extraction_test.py --pdf functional_description.pdf --model qwen3:30b-a3b
```

## Results

| Model | CO2 rule (1.1.10) | Occupancy rule (1.1.13) | Notes |
|---|---|---|---|
| llama3.2:1b | unreliable | unreliable | wrong thresholds, trigger/controlled parameters backwards on the harder rule |
| qwen2.5:7b | correct | correct | one cosmetic `rule_id` naming miss |
| qwen2.5:32b | 6/6 | 5/6 | same accuracy as qwen2.5:7b — see `results/extraction_test_qwen32b.json` |
| qwen3:30b-a3b | 6/6 | 5/6 | same accuracy as qwen2.5:7b, after fixing its hybrid-reasoning mode — see `results/extraction_test_qwen3_30b.json` |

Full raw output (including the model's reasoning, where applicable) is in
`results/`. Raw files for llama3.2:1b and qwen2.5:7b weren't kept from the
earlier test runs — everything from qwen2.5:32b onward is preserved in full.

The "5/6" on the occupancy rule for both qwen2.5:32b and qwen3:30b-a3b is the
same cosmetic miss both times: the scorer checks for a specific keyword in
`rule_id`, and both models named it something else (`"1.1.13"` /
`"OCCUPANCY_SETPOINT_RULE"`) while getting every substantive field right
(trigger, controlled parameter, both branches, correct thresholds). Not a
real extraction error.

### Key finding

**Bigger didn't mean better.** qwen2.5:32b scored identically to qwen2.5:7b
despite far higher compute/VRAM cost. qwen3:30b-a3b (a mixture-of-experts
model) also matched it, but only after specifically handling its "hybrid
reasoning" behavior — by default it burns its output budget on an internal
`<think>` block and returns nothing, so `num_predict`/`num_ctx` needed to be
raised (see `run_test()` in the script) to let it finish reasoning before
writing the answer.

**Recommendation for the pipeline: qwen2.5:7b.** It matches the larger
models' accuracy without the extra hardware cost or the reasoning-mode
handling qwen3 needs.

## Scoring caveat

The scorer (`score_against_ground_truth`) is a simple field-presence /
value-match check, not a semantic one. It's good enough to catch clearly
wrong extractions (e.g. llama3.2:1b's backwards parameters) but can be fooled
— a model can score well on structure while getting meaning wrong. Treat the
scores as a first-pass signal, not a certified accuracy number.
