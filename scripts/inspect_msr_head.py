from pathlib import Path
p = Path('data/MSR_data_cleaned.json')
print('exists', p.exists(), 'size', p.stat().st_size)
with p.open('rb') as f:
    head = f.read(10_000_000)  # 10MB

tokens = [b'[{', b'\n', b'"id"', b'"func_before"', b'"func_after"', b'vul']
for token in tokens:
    idx = head.find(token)
    print(token, 'found at', idx)
    if idx != -1:
        # extract nearest object boundaries
        import re
        start = head.rfind(b'{', 0, idx)
        end = head.find(b'}', idx)
        if start != -1 and end != -1:
            snippet = head[start:end+1]
        else:
            snippet = head[max(0, idx-200):min(len(head), idx+200)]
        print('--- snippet ---')
        print(snippet.decode('utf-8', errors='ignore'))
        print('--- end ---')
        break
