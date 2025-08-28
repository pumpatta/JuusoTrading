import os, sys
print('cwd=', os.getcwd())
print('sys.path (first 10):')
for i,p in enumerate(sys.path[:10]):
    print(i, p)
try:
    import engine
    print('engine imported ok:', engine.__file__)
except Exception as e:
    print('engine import error:', type(e).__name__, e)
