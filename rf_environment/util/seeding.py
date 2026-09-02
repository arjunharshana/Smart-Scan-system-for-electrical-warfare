from __future__ import annotations

import hashlib


def derive_seed(master: int, *parts: object) -> int:
    payload = f"{master}:" + ":".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)
