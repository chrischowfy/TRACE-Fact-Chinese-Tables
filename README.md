# TRACE-Fact

Data and evaluation outputs for *TRACE-Fact: A Program-Grounded Benchmark with Traceable Evidence for Chinese Table Fact Verification*.

Each record is a Chinese claim, one to four Chinese Wikipedia tables and a label: `SUPPORTS`, `REFUTES` or `NEI`
(the given tables are not enough to decide). The label is produced by an executable program stored with the record,
together with the table cells the program reads.

| | |
|---|---|
| Claims | 2,779 |
| Labels | SUPPORTS 1,058 · REFUTES 1,304 · NEI 417 |
| Tables per claim | 1: 900 · 2: 1,157 · 3: 572 · 4: 150 |
| Reasoning groups / skeletons | 14 / 45 |
| Source | Chinese Wikipedia, 350 pages |

## Files

- `data/claims.jsonl`: one record per line.
- `data/table_snapshots.jsonl`: the tables, with page and revision.
- `data/execution_traces.jsonl`: the value of every program step.
- `data/skeleton_registry.jsonl`: the program skeletons and the source-dataset programs that support them.
- `data/quarantine.jsonl`, `data/repairs.jsonl`, `data/known_issues.jsonl`: removed records, records repaired
  without a label change, and wording issues we know of.
- `data/stats.json`: counts by label, group, skeleton, domain and topology.
- `predictions/`: raw model outputs and run settings. `results/results.json`: scores.
- `annotation/`: guideline, sample and judgments for the two human annotation samples.
- `code/eval/`: prompts, agents and scoring.

A record looks like this (tables, program and evidence omitted):

```json
{"id": "r9-01033", "claim": "《八千米以上山峰列表》中，位于巴基斯坦中国的山峰死亡人数合计144人。", "label": "REFUTES"}
```

The other fields are `tables`, `program` (skeleton, operators, slots), `evidence_cells` and `context_cells` (the
cells the program reads), `table_topology` and `quality_flags` (skeleton card, perturbation, structural hardness).
Ids starting with `r9-` are unchanged from the earlier 1,796-record release; `r10-` records were added.

A ratio, percentage or average in a claim is written to a fixed number of decimals. Round the value computed from
the tables half up to the same number of decimals before comparing.

## Usage

```python
import json

with open("data/claims.jsonl", encoding="utf-8") as f:
    claims = [json.loads(line) for line in f]
```

Give a model the claim and its tables and compare its answer with `label`. Keep `program` and the cell fields
hidden unless you run an oracle setting. To score a prediction file:

```bash
python code/eval/score.py --claims data/claims.jsonl --pred flash=predictions/deepseek-chat__full-table/predictions.jsonl
```

## Results

Full tables, thinking off, label-first prompt. Invalid outputs and failed calls count as errors.

| Model or protocol | Acc. | Macro-F1 |
|---|---|---|
| Gemini-3-Flash | 84.6 | 86.1 |
| DeepSeek-V4-Pro | 74.1 | 78.6 |
| GLM-5.1 | 68.4 | 74.0 |
| DeepSeek-V4-Flash | 69.5 | 71.4 |
| Claude-Haiku-4.5 | 63.0 | 68.2 |
| GPT-5.4-mini | 59.9 | 64.2 |
| Qwen3-14B | 51.9 | 53.6 |
| Qwen3-8B | 50.3 | 49.9 |
| Text-to-SQL (DeepSeek-V4-Flash) | 83.2 | 83.1 |
| ReAcTable-style agent (DeepSeek-V4-Flash) | 95.3 | 95.1 |
| Chain-of-Table-style agent (DeepSeek-V4-Flash) | 46.6 | 56.2 |

Only the full-table setting was run for models other than DeepSeek. Claim-only and oracle runs are in
`predictions/` as well.

## Notes

NEI is relative to the given tables, not to outside knowledge. Evidence cells are the cells a program depends on,
not a minimal proof.

The sentence on rounding was added to the prompt on 2026-10-07. In each DeepSeek run 2,202 responses were
requested before that date; the request time is stored with every response.

Two annotators (graduate students, native speakers of Chinese) labeled two samples of 300 records. On the first
they agree on 291 of 299 labels (Cohen's κ = 0.96). On the second, drawn
from the added records, both give the released label on all 300; their notes and cell coordinates
were compiled with software assistance and no time was recorded. 9 of these records were later
removed and 25 reworded.

## License

Data: [CC BY-SA 4.0](LICENSE); keep the Wikipedia URLs and revision IDs when redistributing. Code:
[MIT](LICENSE-CODE).

## Citation

See [CITATION.cff](CITATION.cff).
