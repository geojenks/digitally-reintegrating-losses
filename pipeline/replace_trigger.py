import re
import glob
import os

yaml_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml"))

for path in yaml_files:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'trigger_word:\s*"([^"]+)"', content)
    if not match:
        print(f"SKIP (no trigger_word found): {os.path.basename(path)}")
        continue

    trigger = match.group(1)
    count = content.count("[trigger]")
    if count == 0:
        print(f"SKIP (no [trigger] instances): {os.path.basename(path)}")
        continue

    new_content = content.replace("[trigger]", trigger)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"OK: {os.path.basename(path)} — replaced {count}x [trigger] with '{trigger}'")
