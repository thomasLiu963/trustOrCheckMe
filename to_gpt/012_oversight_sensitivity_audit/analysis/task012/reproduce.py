from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task012_reproduce import reproduce, write_reproduction_md

if __name__ == '__main__':
    result = reproduce()
    write_reproduction_md(result)
    print(result['passed'], result.get('stop_token'))
