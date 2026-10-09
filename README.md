# TRACE-Fact

Data and evaluation outputs for *TRACE-Fact: A Program-Grounded Benchmark with Traceable Evidence for Chinese Table Fact Verification*.

Each record is a Chinese claim, one to four Chinese Wikipedia tables and a label: `SUPPORTS`, `REFUTES` or `NEI`
(the given tables are not enough to decide). The label is produced by an executable program stored with the record,
together with the table cells the program reads.

| | |
|---|---|
| Claims | 4,279 (2,779 built first, 1,500 added afterwards) |
| Labels | SUPPORTS 1,596 · REFUTES 1,833 · NEI 850 |
| Tables per claim | 1: 2,400 · 2: 1,157 · 3: 572 · 4: 150 |
| Reasoning groups / skeletons | 15 / 47 |
| Source | Chinese Wikipedia, 545 pages |

The last 1,500 records (ids `r11-` and `r12-`) were added after the first evaluation to probe agents that query
the tables. 634 state a count, a sum or a mean over a table that ends in a total row, which is not one of the
listed entities. The other 866 come in pairs over one table: one sentence names the edition that is
given, the other names another edition and is `NEI`. Both kinds were chosen after a pilot in which an agent failed on
them, and the 1,126 records with `r12-` ids were added by the same rules after the first
374 had been scored. They are narrow in topic (118 of their 124
total-row tables are medal tables) and were not labeled by the human annotators; `results/results.json` gives every
score on all records, on the 2,779 records built first and on these 1,500.

## Files

- `data/claims.jsonl`: the records, one per line, in release order.
- `data/table_snapshots.jsonl`: the tables, with page and revision.
- `data/execution_traces.jsonl`: the value of every program step.
- `data/skeleton_registry.jsonl`: the program skeletons and the source-dataset programs that support them.
- `data/quarantine.jsonl`, `data/repairs.jsonl`, `data/known_issues.jsonl`: removed records, records repaired
  without a label change, and wording issues we know of.
- `data/stats.json`: counts by label, group, skeleton, domain and topology.
- `predictions/`: raw model outputs and run settings. `results/results.json`: scores on all records and on the
  two parts.
- `annotation/`: guideline, sample and judgments for the two human annotation samples.
- `code/eval/`: prompts, agents and scoring.

A record looks like this (tables, program and evidence omitted):

```json
{"id": "r9-01033", "claim": "《八千米以上山峰列表》中，位于巴基斯坦中国的山峰死亡人数合计144人。", "label": "REFUTES"}
```

The other fields are `tables`, `program` (skeleton, operators, slots), `evidence_cells` and `context_cells` (the
cells the program reads), `table_topology` and `quality_flags` (skeleton card, perturbation, structural hardness).
Ids starting with `r9-` and `r10-` are the 2,779 records of the first build, unchanged; `r11-` and `r12-` are the
records added afterwards.

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

Full tables, thinking off, label-first prompt. Invalid outputs and failed calls count as errors. Accuracy and
macro-F1 are on all records; the last two columns are accuracy on the 2,779 records built first and on
the 1,500 added afterwards.

| Model or protocol | Acc. | Macro-F1 | First | Added |
|---|---|---|---|---|
| Gemini-3-Flash | 88.7 | 89.8 | 84.6 | 96.1 |
| DeepSeek-V4-Pro | 75.3 | 78.4 | 74.1 | 77.6 |
| DeepSeek-V4-Flash | 74.1 | 76.0 | 69.5 | 82.7 |
| GLM-5.1 | 71.8 | 75.5 | 68.4 | 78.1 |
| Claude-Haiku-4.5 | 67.8 | 72.1 | 63.0 | 76.7 |
| GPT-5.4-mini | 64.0 | 67.5 | 59.9 | 71.5 |
| Qwen3-14B | 52.3 | 52.5 | 51.9 | 53.1 |
| Qwen3-8B | 47.0 | 45.3 | 50.3 | 40.8 |
| Text-to-SQL (DeepSeek-V4-Flash) | 73.6 | 76.4 | 83.2 | 55.8 |
| ReAcTable-style agent (DeepSeek-V4-Flash) | 86.1 | 87.8 | 97.0 | 65.8 |
| ReAcTable-style agent (Gemini-3-Flash) | 76.8 | 76.7 | 96.9 | 39.7 |
| Chain-of-Table-style agent (DeepSeek-V4-Flash) | 49.0 | 57.6 | 46.6 | 53.5 |

Only the full-table setting and the ReAcTable-style agent were run for models other than DeepSeek.
Claim-only and oracle runs are in `predictions/` as well. The agent is `code/eval/react_agent.py`. An earlier
harness (`sql_agents.py`) showed six rows of a query result without saying that more existed; it scored 95.3 on the
first 2,779 records (`predictions/deepseek-chat__reactable`).

## Notes

NEI is relative to the given tables, not to outside knowledge. Evidence cells are the cells a program depends on,
not a minimal proof.

The sentence on rounding was added to the prompt on 2026-10-07. In each DeepSeek run made before the extension
2,202 responses were requested before that date; the request time is stored with every response.

Two annotators (graduate students, native speakers of Chinese) labeled two samples of 300 of the first 2,779 records. On the first
they agree on 291 of 299 labels (Cohen's κ = 0.96). On the second both give
the released label on all 300; their notes and cell coordinates were compiled with software assistance
and no time was recorded. 9 of these records were later removed and 25 reworded.
The 1,500 records added afterwards were not labeled by the annotators; they were checked by program replay
and a blind model audit (99.7% agreement on their final wording).

In the 2,779-record release two instance counts of `data/skeleton_registry.jsonl` (NB_LOOKUP2, RJ_MEMBER_RANK) gave
the total of the card the two skeletons share; they are now counted per skeleton.

## License

Data: [CC BY-SA 4.0](LICENSE); keep the Wikipedia URLs and revision IDs when redistributing. Code:
[MIT](LICENSE-CODE).

## Citation

See [CITATION.cff](CITATION.cff).
