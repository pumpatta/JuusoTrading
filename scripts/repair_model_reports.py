import json, os, glob, traceback
from pathlib import Path

reports = glob.glob('models/*.json')

candidates_ext = ['.model', '.trained', '.pkl', '.joblib', '.bin']

changed = []

for r in reports:
    with open(r, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # prefer explicit model_file
    key = None
    model_ref = data.get('model_file') or data.get('model') or data.get('model_path')
    if not model_ref:
        print(f'{r}: no model reference found, skipping')
        continue
    # build candidate paths
    base = Path('models')
    model_path = Path(model_ref)
    if not model_path.is_absolute():
        model_path = base / model_path
    candidates = [model_path]
    # add variations
    stem = model_path.stem
    for ext in candidates_ext:
        candidates.append(base / f"{stem}{ext}")
    # also try scanning for files with stem in name
    for p in base.iterdir():
        if stem.lower() in p.name.lower() and p not in candidates:
            candidates.append(p)
    loaded = False
    loaded_path = None
    load_info = ''
    for cand in candidates:
        try:
            if not cand.exists():
                continue
            # try pickle
            import pickle
            try:
                with open(cand, 'rb') as f:
                    pickle.load(f)
                loaded = True
                loaded_path = cand
                load_info = 'pickle'
                break
            except Exception as e_pickle:
                # try joblib
                try:
                    import joblib
                    joblib.load(cand)
                    loaded = True
                    loaded_path = cand
                    load_info = 'joblib'
                    break
                except Exception as e_joblib:
                    # try xgboost
                    try:
                        import xgboost as xgb
                        bst = xgb.Booster()
                        bst.load_model(str(cand))
                        loaded = True
                        loaded_path = cand
                        load_info = 'xgboost'
                        break
                    except Exception as e_xgb:
                        # try torch for .model which may be a torch zip
                        try:
                            import torch
                            torch.load(str(cand), map_location='cpu')
                            loaded = True
                            loaded_path = cand
                            load_info = 'torch'
                            break
                        except Exception as e_torch:
                            # collect last exception for debugging
                            last_exc = (e_pickle, e_joblib, e_xgb, e_torch)
                            # continue to next candidate
                            continue
        except Exception as e:
            print(f'Unexpected error while testing {cand}: {e}')
            continue
    if loaded:
        # update report to include explicit model_file
        # ensure we pass a string path to relpath
        rel = os.path.relpath(str(loaded_path))
        data['model_file'] = rel
        with open(r, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        changed.append((r, rel, load_info))
        print(f'Updated {r} -> {rel} (via {load_info})')
    else:
        print(f'No loadable candidate found for {r} (tried {len(candidates)} files)')

print('\nSummary:')
for c in changed:
    print(c)
