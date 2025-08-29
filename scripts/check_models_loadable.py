import json
import os
import glob
import pickle
import traceback

reports = glob.glob('models/*.json')
results = []

for r in reports:
    try:
        with open(r, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        results.append((r, False, f"Failed to read JSON: {e}"))
        continue
    model_file = data.get('model_file') or data.get('model') or data.get('model_path')
    if not model_file:
        results.append((r, False, 'No model_file field in report'))
        continue
    # normalize path
    model_path = os.path.normpath(model_file)
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(model_path):
        results.append((r, False, f'Model file missing: {model_path}'))
        continue
    # attempt to load
    load_ok = False
    err = ''
    try:
        # try pickle
        with open(model_path, 'rb') as mf:
            obj = pickle.load(mf)
        load_ok = True
        err = 'loaded with pickle'
    except Exception as e_pickle:
        # try joblib
        try:
            import joblib
            joblib.load(model_path)
            load_ok = True
            err = 'loaded with joblib'
        except Exception as e_joblib:
            # try xgboost native
            try:
                import xgboost as xgb
                bst = xgb.Booster()
                bst.load_model(model_path)
                load_ok = True
                err = 'loaded with xgboost.Booster'
            except Exception as e_xgb:
                # try torch (PyTorch saved models)
                try:
                    import torch
                    torch.load(model_path, map_location='cpu')
                    load_ok = True
                    err = 'loaded with torch.load'
                except Exception as e_torch:
                    err = f'pickle_err={e_pickle}; joblib_err={e_joblib}; xgb_err={e_xgb}; torch_err={e_torch}'
    results.append((r, load_ok, err))

print('Model load report:')
for r, ok, info in results:
    status = 'OK' if ok else 'FAIL'
    print(f'{r}: {status} -> {info}')

# summary
ok_count = sum(1 for _r,o,_i in results if o)
print(f"\nSummary: {ok_count}/{len(results)} model reports had loadable model binaries")
if ok_count < len(results):
    print('Check missing model files or format mismatches (pickle vs xgboost native).')
