import re, pathlib
p = pathlib.Path('Train.py')
t = p.read_text(encoding='utf-8')

# Import swap: torch.cuda.amp -> torch.amp
t = t.replace('from torch.cuda.amp import autocast, GradScaler',
              'from torch.amp import autocast, GradScaler')

# GradScaler(...) -> GradScaler(device_type=..., enabled=...)
t = re.sub(r'\bscaler\s*=\s*GradScaler\(\)',
           'scaler = GradScaler(device_type=device_type, enabled=(device_type == "cuda"))',
           t)

# with autocast(): -> with autocast(device_type, enabled=(device_type == "cuda")):
t = re.sub(r'with\s+autocast\(\):',
           'with autocast(device_type, enabled=(device_type == "cuda")):',
           t)

# Ensure device_type is defined after your 'device = ...' line
if 'device_type' not in t:
    t = re.sub(r'(device\s*=\s*"cuda"\s*if\s*torch\.cuda\.is_available\(\)\s*else\s*"cpu")',
               r'\1\ndevice_type = "cuda" if device == "cuda" else "cpu"',
               t, count=1)

p.write_text(t, encoding='utf-8')
print("Patched Train.py successfully.")
