import os, sys, io, json, ast, shutil, datetime as dt
import numpy as np

# insert root directory into python module search path
sys.path.insert(1, os.getcwd())
from utils import npz_file_management

readpath = "data/wfss_data/Algarve_Est_Example_File_1.wfss"
writepath = "data/wfss_data/Algarve_Est_Example_File_1_edited2.5.wfss"

# ---------- helpers ----------

def _parse_bool(s: str) -> bool:
    s2 = s.strip().lower()
    if s2 in {"true", "t", "1", "yes", "y"}: return True
    if s2 in {"false", "f", "0", "no", "n"}: return False
    raise ValueError(f"Cannot parse boolean from '{s}'")

def coerce_to_type(value_str: str, prototype):
    """Convert input to the same type as prototype."""
    if isinstance(prototype, (np.floating, float)):
        return type(prototype)(float(value_str))
    if isinstance(prototype, (np.integer, int)):
        return type(prototype)(int(value_str))
    if isinstance(prototype, (np.bool_, bool)):
        return _parse_bool(value_str)
    if isinstance(prototype, str):
        return value_str
    if isinstance(prototype, np.ndarray):
        s = value_str.strip()
        try:
            arr = np.array(json.loads(s))  # try JSON list
        except Exception:
            parts = [p for p in s.replace(",", " ").split() if p]
            arr = np.array([ast.literal_eval(p) if p[0] in "[{(" else float(p) for p in parts])
        return arr.astype(prototype.dtype, copy=False)
    return ast.literal_eval(value_str)

def unwrap_dicts(data):
    """Convert 0-D object arrays containing dicts into real dicts."""
    for k, v in list(data.items()):
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                item = v.item()
                if isinstance(item, dict):
                    data[k] = item
            except Exception:
                pass
    return data

def wrap_dicts(data):
    """Convert dicts back into 0-D object arrays for saving."""
    arrays = {}
    for k, v in data.items():
        if isinstance(v, dict):
            arrays[k] = np.array(v, dtype=object)
        else:
            arrays[k] = v
    return arrays

# ---------- read ----------

data = npz_file_management.read_and_store_npz_contents(readpath)
data = unwrap_dicts(data)

print("Loaded data. Example groups:")
print(" - unmod_settings")
print(" - mod_settings")
print(" - metadata")
print("Use syntax: group.var=value (e.g., unmod_settings.cell_size_x=80)\n")

# ---------- edit loop ----------

while True:
    line = input("Edit> ").strip()
    if line.lower() == "done":
        break
    if line.lower() == "show":
        for g, obj in data.items():
            print(f"{g}: {obj}")
        continue

    if "=" not in line or "." not in line.split("=", 1)[0]:
        print("Use group.var=value or 'show' or 'done'")
        continue

    left, value_str = line.split("=", 1)
    group, var = left.split(".", 1)
    group, var = group.strip(), var.strip()

    if group not in data or not isinstance(data[group], dict):
        print(f"[error] Unknown group: {group}")
        continue
    if var not in data[group]:
        print(f"[error] Unknown variable in {group}: {var}")
        continue

    old_value = data[group][var]
    try:
        new_value = coerce_to_type(value_str.strip(), old_value)
        data[group][var] = new_value
        print(f"Updated {group}.{var}: {old_value!r} -> {new_value!r}")
    except Exception as e:
        print(f"[error] Could not update: {e}")

# ---------- save ----------

arrays = wrap_dicts(data)
blob = npz_file_management.build_npz_file(arrays)
with open(writepath, "wb") as f:
    f.write(blob)

print(f"\n[DONE] File saved as: {writepath}")
