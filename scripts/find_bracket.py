from secure_pipeline.convert_bigvul import _open_text
from pathlib import Path
p=Path('data/MSR_data_cleaned.json')
with _open_text(p) as f:
    data=f.read(500000)
    idx=data.find('[')
    print('idx',idx)
    if idx!=-1:
        print(data[idx:idx+300])
    else:
        print('no bracket in first 500k')
