"""Finish failed audit stages without replacing valid emitted labels or base records.

Deterministic eligibility: every incomplete v2/all300 record. Preserve a
parseable first-stage judgment even if table-necessity applicability is wrong;
flag that field instead. Retry an unparseable first stage once, then obtain the
missing evidence judgment. Supplements never overwrite the original JSONL.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from model_pre_audit import OUT, C, material, payload, validate
from diagnose_revision_preview import normalize_output

RUN=OUT/'deepseek-v4-pro/v2/all300'
RETRY_SUFFIX='\n补充输出约束：上一轮未得到可解析的完整JSON。请重新独立判断，note最多80字、evidence_coordinates最多3处关键坐标、issues最多2项；不要重复句子或逐行列举，必须输出所有必需字段并闭合JSON。'


def relaxed_validate(j, claim):
    """Validate content without laundering a wrong applicability field into NA."""
    shadow=dict(j)
    warnings=[]
    not_applicable=len(claim['tables'])==1 or j.get('ann_label') in ('NEI','UNCERTAIN')
    if not_applicable and j.get('needs_all_tables')!='NA':
        warnings.append('table_necessity_not_applicable_but_non_NA_returned')
        shadow['needs_all_tables']='NA'
    elif not not_applicable and j.get('needs_all_tables')=='NA':
        warnings.append('table_necessity_not_assessed')
    # Check the original field independently even when its applicability is wrong.
    assert str(j['needs_all_tables']) in ('1','0','U','NA')
    validate(shadow,claim)
    return warnings


def main():
    claims,key,_=material(); lookup={k['item']:k for k in key}
    source=RUN/'judgments.jsonl'
    rows=list(map(json.loads,source.read_text().splitlines()))
    assert len(rows)==len({r['item'] for r in rows})==300
    config=json.loads((RUN/'run_meta.json').read_text())
    parent_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    policy={'parent_sha256':parent_hash,'model':config['model'],'retries_per_missing_stage':1,
            'eligibility':'all incomplete first-pass records, independent of labels/correctness',
            'label_retry_suffix':RETRY_SUFFIX,
            'parsed_labels_preserved':True,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=RUN/'completion_supplement.jsonl'; meta_path=RUN/'completion_meta.json'
    if meta_path.exists(): assert json.loads(meta_path.read_text())==policy
    else: meta_path.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    existing=list(map(json.loads,out.read_text().splitlines())) if out.exists() else []
    done={r['item'] for r in existing}; assert len(done)==len(existing)
    key_string=C.api_key('DEEPSEEK_API_KEY')
    def one(base):
        item=base['item']; claim=claims[lookup[item]['id']]
        rec={'item':item,'id':base['id'],'parent_sha256':parent_hash,'input_sha256':base['input_sha256'],
             'api_calls':[],'human':False,'status':'incomplete','label_reused':False}
        messages=[{'role':'system','content':config['system']},
                  {'role':'user','content':json.dumps(payload(claim,item),ensure_ascii=False)}]
        raw=base['api_calls'][0]['raw_output']
        try:
            judgment,_=normalize_output(raw)
            warnings=relaxed_validate(judgment,claim)
            rec['label_reused']=True
        except (ValueError,TypeError,KeyError,AssertionError,AttributeError):
            messages[-1]['content']+=RETRY_SUFFIX
            rec['label_retry_input_sha256']=hashlib.sha256(messages[-1]['content'].encode()).hexdigest()
            raw,api=C.call_llm(messages,model=config['model'],base_url='https://api.deepseek.com',api_key=key_string,
                               max_tokens=4096,temperature=0,extra={'thinking':{'type':'disabled'}})
            rec['api_calls'].append({'stage':'label_retry','raw_output':raw,'api':api})
            try:
                assert not api.get('error')
                judgment,_=normalize_output(raw)
                warnings=relaxed_validate(judgment,claim)
            except (ValueError,TypeError,KeyError,AssertionError,AttributeError): return rec
        rec['label_stage_judgment']=judgment
        rec['workflow_warnings']=warnings
        cells=claim['evidence_cells'] or claim['context_cells']
        index={t['table_id']:f't{i}' for i,t in enumerate(claim['tables'])}
        hints=[{'table':index[x['table_id']],'row':x['row'],'column':x['col'],'value':x['raw_value']} for x in cells]
        messages += [{'role':'assistant','content':raw},{'role':'user','content':
            '保持刚才标签不变。现在检查以下依赖提示是否相关且足以核验；NEI需检查完整表格的缺失信息。提示为空或不确定用U；忽略同表矛盾用0。提示不要求最小，不应仅因含竞争项判0。只输出JSON：evidence_ok(1/0/U), evidence_note(中文)。提示：'+json.dumps(hints,ensure_ascii=False)}]
        raw2,api=C.call_llm(messages,model=config['model'],base_url='https://api.deepseek.com',api_key=key_string,
                            max_tokens=2048,temperature=0,extra={'thinking':{'type':'disabled'}})
        rec['api_calls'].append({'stage':'evidence_completion','raw_output':raw2,'api':api})
        try:
            assert not api.get('error')
            evidence,_=normalize_output(raw2)
            assert str(evidence['evidence_ok']) in ('1','0','U')
            rec['judgment']=judgment|{'evidence_ok':str(evidence['evidence_ok']),'evidence_note':evidence.get('evidence_note','')}
            rec['status']='complete'
        except (ValueError,TypeError,KeyError,AssertionError,AttributeError): pass
        return rec
    pending=[r for r in rows if r['status']!='complete' and r['item'] not in done]
    print(f'Completing {len(pending)} incomplete records; original outputs retained',flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor, out.open('a') as output:
        for future in as_completed([executor.submit(one,r) for r in pending]):
            rec=future.result(); output.write(json.dumps(rec,ensure_ascii=False)+'\n'); output.flush()
            print(rec['item'],rec['status'],'label_reused='+str(rec['label_reused']),flush=True)


if __name__=='__main__':main()
