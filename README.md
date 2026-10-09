# TRACE-Fact

Chinese table fact verification. Each record has a Chinese claim, 1 to 4 tables from Chinese Wikipedia and a label
(`SUPPORTS`, `REFUTES` or `NEI`, i.e. not decidable from the given tables). Labels come from a small executable
program that is stored with the record, together with the table cells it reads.

Paper: *TRACE-Fact: A Program-Grounded Benchmark with Traceable Evidence for Chinese Table Fact Verification* (CCL 2026).

## Data

| | |
|---|---|
| Claims | 4,279 |
| Labels | SUPPORTS 1,596 / REFUTES 1,833 / NEI 850 |
| Tables per claim | 1: 2,400 / 2: 1,157 / 3: 572 / 4: 150 |
| Reasoning groups / skeletons | 15 / 47 |
| Source | Chinese Wikipedia, 545 pages |

634 records aggregate (count, sum or mean) over a table that ends with a total row; the REFUTES version is
what you get if you count the total row in. 866 records come in pairs over the same table: one names
the edition that is given, the other names a different edition and is NEI. These are the `r11-*` and `r12-*` ids. We
added them to test agents that query the tables with SQL, so most of them are medal tables (118 of the
124 total-row tables).

Files:

- `data/claims.jsonl` - all records, one per line
- `data/table_snapshots.jsonl` - the tables, with page and revision
- `data/execution_traces.jsonl` - the value of every program step
- `data/skeleton_registry.jsonl` - program skeletons and the source-dataset programs behind them
- `data/rewording.jsonl` - claims reworded on 2026-10-09, old and new text
- `data/quarantine.jsonl`, `data/repairs.jsonl`, `data/known_issues.jsonl` - removed records, rule-based repairs
  (labels unchanged), wording issues we know about
- `data/stats.json` - counts by label, group, skeleton, domain and topology
- `predictions/` - raw model outputs and run settings; `results/results.json` - scores
- `code/eval/` - prompts, agents, scoring

Example (tables, program and evidence left out):

```json
{"id": "r9-00088", "claim": "冬奥会奖牌榜上，挪威2014年的金牌数比2010年增加了1枚。", "label": "REFUTES"}
```

Other fields: `tables`, `program` (skeleton, operators, slots), `evidence_cells`, `context_cells`, `table_topology`,
`quality_flags`.

Ratios, percentages and averages in the claims have a fixed number of decimals. Round your computed value (half up)
to the same number of decimals before comparing.

## Usage

```python
import json

with open("data/claims.jsonl", encoding="utf-8") as f:
    claims = [json.loads(line) for line in f]
```

Show the model the claim and the tables, compare the answer with `label`. Don't show `program` or the cell fields
unless you run an oracle setting. Scoring:

```bash
python code/eval/score.py --claims data/claims.jsonl --pred flash=predictions/deepseek-chat__full-table/predictions.jsonl
```

## Results

Full tables, thinking off, label-first prompt. Invalid outputs and failed API calls count as wrong. All 4,279 records.

| Model / protocol | Acc. | Macro-F1 |
|---|---|---|
| Gemini-3-Flash | 88.7 | 89.8 |
| DeepSeek-V4-Pro | 75.3 | 78.4 |
| DeepSeek-V4-Flash | 74.1 | 76.0 |
| GLM-5.1 | 71.8 | 75.5 |
| Claude-Haiku-4.5 | 67.8 | 72.1 |
| GPT-5.4-mini | 64.0 | 67.5 |
| Qwen3-14B | 52.3 | 52.5 |
| Qwen3-8B | 47.0 | 45.3 |
| Text-to-SQL (DeepSeek-V4-Flash) | 73.6 | 76.4 |
| ReAcTable-style agent (DeepSeek-V4-Flash) | 86.1 | 87.8 |
| ReAcTable-style agent (Gemini-3-Flash) | 76.9 | 76.8 |
| Chain-of-Table-style agent (DeepSeek-V4-Flash) | 49.0 | 57.6 |

Models other than DeepSeek were only run in the full-table setting (plus the agent with Gemini).
Claim-only and oracle runs are in `predictions/` too. The agent is `code/eval/react_agent.py`. An earlier version
(`sql_agents.py`) only showed 6 rows of each query result without saying there were more; it got 83.2
(`predictions/deepseek-chat__reactable`).

## License

Data (`data/`): [CC BY-SA 4.0](LICENSE). The tables come from Chinese Wikipedia, which is CC BY-SA too, so
share-alike carries over. If you redistribute, keep each table's `source.url` and `source.revision_id` so the page
and revision can be credited. Code (`code/`): [MIT](LICENSE-CODE).

## Citation

See [CITATION.cff](CITATION.cff).
