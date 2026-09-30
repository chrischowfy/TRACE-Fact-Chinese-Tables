"""Reconstruct the selected r7 candidate from table snapshots and program plans.

No claims.jsonl content is used to rebuild labels or evidence. This is a replay
of the published selection, not recovery of the lost reviewed data/derivations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from prepare_review_r7 import read_jsonl, hydrate
from r7_extensions import replay
from repair_review_release import file_hash


def reconstruct(release,out=None):
    from r7_runtime import enable
    enable()
    manifest=json.loads((release/'snapshot_manifest.json').read_text())
    for name,digest in manifest['artifacts_sha256'].items():
        if file_hash(release/name)!=digest:
            raise ValueError('Artifact changed: '+name)
    tables={t['table_id']:t for t in read_jsonl(release/'table_snapshots.jsonl')}
    rebuilt=[]
    for compact in read_jsonl(release/'selection.jsonl'):
        row=hydrate(compact,tables)
        result=replay(row)
        if row['label']!=result['label']: raise ValueError('Label replay mismatch: '+row['id'])
        row['label']=result['label']
        row['evidence_cells']=result['cells'] if result['label']!='NEI' else []
        row['context_cells']=result['cells'] if result['label']=='NEI' else []
        rebuilt.append(row)
    data=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rebuilt).encode()
    actual=hashlib.sha256(data).hexdigest()
    if actual!=manifest['claims_sha256']:
        raise ValueError('Reconstructed bytes do not match candidate snapshot')
    if out:
        if out.exists(): raise ValueError('Replay target must not exist')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    return {'records':len(rebuilt),'replayed_sha256':actual,'byte_identical':True,
            'api_calls':0,'selection_replay_not_original_artifact_recovery':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    print(json.dumps(reconstruct(args.release,args.out),indent=2))
