from pathlib import Path
import sys

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    paths = [Path('models/B_TCN.model'), Path('models/B_TCN.trained'), Path('models/xgb_anchor.trained'), Path('models/xgb_anchor.txt')]

for p in paths:
    print('---')
    print('Path:', p)
    print('Exists:', p.exists())
    if not p.exists():
        continue
    try:
        size = p.stat().st_size
        print('Size:', size)
        with p.open('rb') as f:
            head = f.read(256)
        print('Head (repr):', repr(head[:128]))
        try:
            text = head.decode('utf-8', errors='replace')
            print('Head (utf8):', text[:256])
        except Exception as e:
            print('Decode error:', e)
    except Exception as e:
        print('Error reading file:', e)
