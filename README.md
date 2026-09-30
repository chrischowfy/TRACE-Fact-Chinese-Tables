# TRACE-Fact

TRACE-Fact: A Program-Grounded Benchmark with Traceable Evidence for Chinese Table Fact Verification

TRACE-Fact verifies Chinese claims against Chinese Wikipedia tables. Each claim is labeled `SUPPORTS`, `REFUTES` or
`NEI` (not enough information in the given tables). Every record includes the source tables, the executable program
that determines the label, and the table cells that the program reads.

## Statistics

| | |
|---|---|
| Claims | 1,796 |
| Labels | SUPPORTS 708 · REFUTES 818 · NEI 270 |
| Multi-table inputs | 968 (53.9%) |
| Tables per claim | 1 → 828 · 2 → 689 · 3 → 275 · 4+ → 4 |
| Reasoning groups / skeletons | 6 / 24 |
| Near-miss / runner-up refutations | 437 / 80 |
| NEI with a withheld table | 162 |
| Source pages | 316 |
| Source | Chinese Wikipedia |
| Data license | CC BY-SA 4.0 |

## Data

`data/claims.jsonl` stores one record per line. A record looks like this (tables, program and evidence omitted):

```json
{"id": "r9-01033", "claim": "《八千米以上山峰列表》中，位于巴基斯坦中国的山峰死亡人数合计144人。", "label": "REFUTES"}
```

The main fields are:

- `claim`, `label`: the Chinese statement and its three-way label.
- `tables`: headers, rows and source (page, revision ID, URL) of each given table.
- `program`: the reasoning skeleton, its operators and bound arguments.
- `evidence_cells`, `context_cells`: cells read by the program, including compared competitors.
- `table_topology`, `quality_flags`: input structure, structural hardness and per-table checks.

`data/execution_traces.jsonl` gives the value produced by each program step on the given tables, for example
`类别06 → [乔戈里峰, 加舒尔布鲁木I峰, 布洛阿特峰, 加舒尔布鲁木II峰] → [81, 29, 21, 21] → 152 → false` for the record above.
The other files in `data/` are the table snapshots, the admitted skeleton registry, per-table necessity witnesses,
statistics, and the lists described under Notes. `registry/` contains the skeleton induction from verified
TabFact, WikiTableQuestions, TAT-QA and MultiModalQA programs.

## Usage

```python
import json

with open("data/claims.jsonl", encoding="utf-8") as f:
    claims = [json.loads(line) for line in f]

print(len(claims))
print(claims[0]["claim"], claims[0]["label"])
```

For verification, give a model the claim and its tables and compare the predicted label with `label`. Keep the
program and evidence fields hidden unless you run an oracle setting.

## Evaluation

Results of non-thinking DeepSeek models with a label-first prompt (accuracy / macro-F1; invalid outputs count as errors):

| Protocol | Acc. | Macro-F1 |
|---|---|---|
| Full table, DeepSeek-V4-Flash | 69.4 | 70.6 |
| Full table, DeepSeek-V4-Pro | 75.9 | 79.7 |
| Text-to-SQL (Flash) | 82.4 | 81.4 |
| ReAcTable-style agent (Flash) | 94.4 | 93.6 |
| Chain-of-Table-style agent (Flash) | 46.7 | 58.5 |

`predictions/` holds the raw outputs, `results/` the scores and diagnostics, `audit/` the construction-time model
audit, and `annotation/` the human annotation materials. To score a prediction file:

```bash
python code/eval/score.py --claims data/claims.jsonl --pred flash=predictions/deepseek-chat__full-table/predictions.jsonl
```

The evaluation code is in `code/eval/` (full-table, claim-only and oracle prompting, text-to-SQL, ReAcTable-style and
Chain-of-Table-style agents); the construction code is in `code/src/` and `code/tools/`.

## Notes

NEI is defined relative to the given tables, not outside knowledge. Evidence cells record program dependencies
rather than a minimal proof. Two annotators (graduate students, native speakers of Chinese) independently labeled a
stratified sample of 300 records; on the 299 records that remain in the release they agree on 291 labels
(Cohen's κ = 0.96). The annotation showed that one player table was an own-goal list, so the four records that used
it as a scorer list were removed (`data/quarantine.jsonl`). Nineteen REFUTES claims write a decrease with a signed
value (e.g. 下降了-5.4%), and one claim omits a percentage unit; their labels are correct, and they are listed in
`data/known_issues.jsonl`. Tables are snapshots at the cited Wikipedia revisions, and source coverage is curated.

## License

The data is released under [CC BY-SA 4.0](LICENSE); keep the Wikipedia source URLs and revision IDs when
redistributing it. The code is released under the [MIT License](LICENSE-CODE).

## Citation

See [CITATION.cff](CITATION.cff).
