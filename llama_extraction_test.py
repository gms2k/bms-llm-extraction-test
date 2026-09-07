"""
Structured rule extraction test - Germanus

ollama pull qwen2.5:7b   (or llama3.2:1b / qwen2.5:32b / qwen3:30b-a3b)
pip install ollama
pip install docling      (only needed for --pdf)

python llama_extraction_test.py --model qwen2.5:7b
python llama_extraction_test.py --pdf functional_description.pdf --model qwen3:30b-a3b

Asks the model to pull one control rule out of the FD into a fixed JSON
schema (this is Stage 2 - LLM Rule Extraction) and checks it against a
ground truth we hand-verified from the PDF. Two rules, same ones Morshed
used in the week 3 meeting:
  1. CO2 damper control (1.1.10) - easy, proportional
  2. Occupied mode temp setpoint (1.1.13) - harder, branching
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import ollama

CO2_SECTION_TEXT = """1.1.10. CO2 Control
In 'normal operation' the BMS will modulate each VRV zone outside air damper
to control zone CO2. The outside air damper will be at its minimum outside
air damper position until the zone CO2 reaches its minimum setpoint of 600
ppm (adj.) and will open proportionally to a maximum outside air damper
position when the zone CO2 is equal to or greater than 800 ppm (adj.).

Fig. 1.1.9 Sample CO2 Damper Control. In this example, the minimum outside
air damper position is commissioned at 40% open and the maximum outside air
damper position at 80% open. Both parameters are adjustable on the BMS as
required.
"""

OCCUPIED_MODE_SECTION_TEXT = """1.1.13. Occupied Mode Control
There are 14 XOVIS people counting sensors placed at entry and exit points on
each floor of the building and for each floor, the associated sensors' In
Count and Out Count is aggregated.

The Occupancy is calculated as below:
Total Floor In Count - Total Floor Out Count = Floor Occupancy

The "total floor in count" will be used to adjust the thermal condition
setpoints depending on the amount of people recorded to be in the floor.
Specific details are as follows:

Operation Mode (For Individual floors): All hours
Floor Occupancy from Lighting System: ON
HVAC Status: ON
Abakus People Count: >5
Thermal Conditions Type: Narrow
Thermal Conditions description: Fixed setpoint of 22.5C (adj.) with a
deadband of 1.5C (adj.)

Operation Mode (For Individual floors): All hours
Floor Occupancy from Lighting System: ON
HVAC Status: ON
Abakus People Count: <=5
Thermal Conditions Type: Wide
Thermal Conditions description: Variable setpoint of 19C to 25C (adj.) with a
fixed deadband of 1.5C (adj.) depending on ambient air temperatures.

Note: the zone temperature setpoint will scale linearly with building
outside air temperature within the wide-band range.
"""

# hand-checked against the PDF (Rev. H) ourselves
GROUND_TRUTH = {
    "co2_control": {
        "rule_id": "1.1.10_CO2_Control",
        "trigger_parameter": "Zone CO2 (ppm)",
        "controlled_parameter": "Outside Air Damper Position (%)",
        "condition_type": "proportional",
        "thresholds": [
            {"input": 600, "output": 40},
            {"input": 800, "output": 80},
        ],
        "applies_to": "each VRV zone",
    },
    "occupied_mode_temp": {
        "rule_id": "1.1.13_Occupied_Mode_Temperature_Setpoint",
        "trigger_parameter": "Floor Occupancy (people count) and Outside Air Temperature",
        "controlled_parameter": "Zone Temperature Setpoint (deg C)",
        "condition_type": "conditional_branching",
        "branches": [
            {
                "condition": "occupancy > 5",
                "setpoint_type": "fixed",
                "value": 22.5,
                "deadband": 1.5,
            },
            {
                "condition": "occupancy <= 5",
                "setpoint_type": "variable",
                "range": [19, 25],
                "deadband": 1.5,
                "depends_on": "outside air temperature",
            },
        ],
        "applies_to": "each floor",
    },
}

SCHEMA_PROMPT = """You are extracting a control rule from a Building Management System \
functional description document into strict JSON. Read the section below and output \
ONLY a JSON object (no prose, no markdown fences) matching this shape:

{{
  "rule_id": "<short id>",
  "trigger_parameter": "<what is measured/monitored>",
  "controlled_parameter": "<what gets adjusted>",
  "condition_type": "<'proportional' | 'conditional_branching' | 'fixed'>",
  "thresholds": [ {{"input": <number>, "output": <number>}}, ... ]  // omit if not proportional
  "branches": [ {{"condition": "<text>", ...other relevant fields}} ]  // omit if not branching
  "applies_to": "<scope, e.g. 'each VRV zone' or 'each floor'>",
  "source_text": "<the exact sentence(s) you used, quoted from the section>"
}}

If a field does not apply, omit it rather than guessing a value.

SECTION:
\"\"\"
{section_text}
\"\"\"

JSON:"""


def convert_pdf_to_markdown(pdf_path: str) -> str:
    # same approach as nhy's docling script
    from docling.document_converter import DocumentConverter

    print(f"converting {pdf_path} to markdown with docling, this takes a bit...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def extract_section(markdown_text: str, section_number: str) -> str:
    # grabs one numbered section out of the converted markdown, up to the
    # next section heading at the same depth
    pattern = re.compile(
        r"(?:^|\n)"
        r"(#{0,6}\s*" + re.escape(section_number) + r"\.?\s.*?)"
        r"(?=\n#{0,6}\s*\d+(?:\.\d+){2,}\.?\s|\Z)",
        re.DOTALL,
    )
    match = pattern.search(markdown_text)
    if not match:
        raise ValueError(
            f"couldn't find section '{section_number}' in the converted markdown - "
            f"check output.md and adjust --co2-section / --occ-section if docling "
            f"rendered the headings differently"
        )
    return match.group(1).strip()


def extract_json(raw_text: str):
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None, raw_text
    try:
        return json.loads(match.group(0)), raw_text
    except json.JSONDecodeError:
        return None, raw_text


def score_against_ground_truth(extracted: dict, truth: dict) -> dict:
    # naive field-presence / value-match scorer, not exhaustive, just
    # enough to eyeball how close a model got
    if extracted is None:
        return {"score": 0, "total": 1, "notes": ["Model did not return valid JSON"]}

    notes = []
    total = 0
    score = 0

    def check(label, got, expected):
        nonlocal total, score
        total += 1
        if got == expected:
            score += 1
        else:
            notes.append(f"{label}: expected {expected!r}, got {got!r}")

    check("rule_id (contains keyword)", (truth["rule_id"].split("_")[1].lower() in str(extracted.get("rule_id", "")).lower()), True)
    check("trigger_parameter present", bool(extracted.get("trigger_parameter")), True)
    check("controlled_parameter present", bool(extracted.get("controlled_parameter")), True)
    check("condition_type", extracted.get("condition_type"), truth["condition_type"])

    if "thresholds" in truth:
        got_thresh = extracted.get("thresholds")
        check("thresholds match", got_thresh, truth["thresholds"])

    if "branches" in truth:
        got_branches = extracted.get("branches")
        total += 1
        if got_branches and len(got_branches) == len(truth["branches"]):
            score += 1
        else:
            notes.append(f"branches: expected {len(truth['branches'])} branches, got {len(got_branches) if got_branches else 0}")

    check("source_text present", bool(extracted.get("source_text")), True)

    return {"score": score, "total": total, "notes": notes}


# qwen3 thinks through a <think> block before answering by default. we just
# want the final JSON so /no_think skips that unless --think is passed
THINKING_MODEL_PREFIXES = ("qwen3",)


def run_test(model: str, label: str, section_text: str, truth: dict, think: bool = False):
    print(f"\n{'=' * 70}\nTEST: {label}  (model: {model})\n{'=' * 70}")
    prompt = SCHEMA_PROMPT.format(section_text=section_text)

    is_thinking_model = model.lower().startswith(THINKING_MODEL_PREFIXES)
    if is_thinking_model and not think:
        prompt += "\n/no_think"

    options = {"temperature": 0}
    if is_thinking_model:
        # no cap on thinking length, otherwise it runs out of budget mid
        # reasoning on the harder rule and content comes back empty
        options["num_predict"] = -1
        options["num_ctx"] = 16384

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=options,
    )
    raw = response["message"]["content"]
    thinking = response["message"].get("thinking") or ""
    done_reason = response.get("done_reason", "unknown")

    print("--- Raw model output ---")
    print(raw if raw else "(empty)")
    print(f"(done_reason: {done_reason}, thinking length: {len(thinking)} chars)")
    if done_reason == "length":
        print("stopped because it hit the num_ctx/num_predict limit, not because "
              "it finished on its own - bump num_ctx up more if this keeps happening")
    if thinking:
        print("\n--- thinking field (not scored, first/last 1000 chars) ---")
        if len(thinking) > 2000:
            print(thinking[:1000] + "\n...\n" + thinking[-1000:])
        else:
            print(thinking)

    if not raw and thinking:
        print("\ncontent was empty but thinking wasn't - model spent the whole "
              "turn reasoning and never wrote the JSON. trying to pull it out of "
              "the thinking text instead.")
        raw_for_parsing = thinking
    elif not raw and is_thinking_model:
        print("\ncontent and thinking both empty, rerun or pass --think to see "
              "what it's doing")
        raw_for_parsing = raw
    else:
        raw_for_parsing = raw

    parsed, cleaned = extract_json(raw_for_parsing)
    print("\n--- Parsed JSON ---")
    print(json.dumps(parsed, indent=2) if parsed else "(could not parse valid JSON)")

    result = score_against_ground_truth(parsed, truth)
    print(f"\n--- Score: {result['score']}/{result['total']} ---")
    for n in result["notes"]:
        print(f"  - {n}")

    return {
        "label": label,
        "model": model,
        "raw_output": raw,
        "thinking": thinking,
        "parsed": parsed,
        "score": result["score"],
        "total": result["total"],
        "notes": result["notes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--out", default="extraction_test_results.json")
    parser.add_argument("--pdf", default=None, help="path to the FD PDF, pulls sections from the "
                                                      "docling-converted doc instead of hardcoded text")
    parser.add_argument("--out-md", default="output.md", help="where to save the converted markdown (--pdf only)")
    parser.add_argument("--co2-section", default="1.1.10")
    parser.add_argument("--occ-section", default="1.1.13")
    parser.add_argument("--think", action="store_true", help="let qwen3 think instead of /no_think")
    args = parser.parse_args()

    if args.pdf:
        markdown_text = convert_pdf_to_markdown(args.pdf)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"saved converted doc to {args.out_md}")
        co2_text = extract_section(markdown_text, args.co2_section)
        occ_text = extract_section(markdown_text, args.occ_section)
    else:
        co2_text = CO2_SECTION_TEXT
        occ_text = OCCUPIED_MODE_SECTION_TEXT

    results = []
    results.append(run_test(args.model, "Easy rule: CO2 Damper Control (1.1.10)", co2_text, GROUND_TRUTH["co2_control"], think=args.think))
    results.append(run_test(args.model, "Tricky rule: Occupancy+Temp Setpoint (1.1.13)", occ_text, GROUND_TRUTH["occupied_mode_temp"], think=args.think))

    summary = {
        "model": args.model,
        "source": args.pdf if args.pdf else "hardcoded section text",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSUMMARY - model: {args.model}")
    for r in results:
        print(f"  {r['label']}: {r['score']}/{r['total']}")
    print(f"saved full results to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
