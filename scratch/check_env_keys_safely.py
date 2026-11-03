from pathlib import Path

env_path = Path("d:/NLP Project/.env")
print(f".env exists: {env_path.exists()}")

if env_path.exists():
    lines = env_path.read_text(encoding="utf-8").splitlines()
    print("Variables declared in .env:")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            var_name, var_val = line.split("=", 1)
            var_name = var_name.strip()
            val_clean = var_val.strip().strip("'\"")
            is_empty = len(val_clean) == 0
            has_placeholder = "your" in val_clean.lower() or "xxx" in val_clean.lower()
            print(f"  - Variable: '{var_name}' | is_empty: {is_empty} | has_placeholder: {has_placeholder}")
