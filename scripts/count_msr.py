import json
from collections import Counter
from pathlib import Path
p = Path('data/MSR_data_cleaned.json')
if not p.exists():
    print('MISSING')
    raise SystemExit(1)

counts = Counter()
lines = 0
broken = False
with p.open('r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            broken = True
            break
        lines += 1
        vul = str(obj.get('vul') or obj.get('vulnerable') or '')
        if vul not in ('', '0', 'false', 'False'):
            counts['vulnerable'] += 1
        else:
            if obj.get('func_before'):
                counts['seguro'] += 1
            if obj.get('func_after'):
                counts['seguro'] += 1

if broken:
    print('NOT_JSONL')
else:
    print('lines', lines)
    print('vulnerable', counts['vulnerable'])
    print('seguro', counts['seguro'])
