from secure_pipeline.convert_bigvul import _iter_json_array, _iter_jsonl, _open_text
from pathlib import Path
p = Path('data/MSR_data_cleaned.json')
print('size', p.stat().st_size)
# try jsonl
print('Trying JSONL iterator (first 3)')
count=0
for o in _iter_jsonl(p):
    print({k: type(v) for k,v in list(o.items())[:6]})
    count+=1
    if count>=3: break

print('Trying JSON array iterator (first 3)')
count=0
for o in _iter_json_array(p):
    print({k: type(v) for k,v in list(o.items())[:6]})
    count+=1
    if count>=3: break
print('done')
