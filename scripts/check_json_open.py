from secure_pipeline.convert_bigvul import _open_text
from pathlib import Path
p=Path('data/MSR_data_cleaned.json')
out=Path('scripts/_check_open.txt')
with _open_text(p) as f:
    head=f.read(200)
with out.open('w',encoding='utf-8') as o:
    o.write(f'exists:{p.exists()} size:{p.stat().st_size}\n')
    o.write(f'head_len:{len(head)}\n')
    o.write(repr(head[:200]))
print('wrote',out)
