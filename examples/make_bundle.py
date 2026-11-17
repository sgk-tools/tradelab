"""Pack the source files needed to rebuild the app into dist/src.json.

The daily scheduled run restores this bundle, so re-run and republish it
whenever the code changes:  python examples/make_bundle.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted(ROOT.glob("tradelab/*.py")) + [ROOT / "tradelab/app_template.html"]
bundle = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in files}
out = ROOT / "dist/src.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
print(f"{len(bundle)} files -> {out} ({out.stat().st_size:,} bytes)")
