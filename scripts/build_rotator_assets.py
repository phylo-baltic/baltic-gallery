#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "source" / "_static"
OUTPUT = STATIC_ROOT / "rotator-images.js"


def collect_png_paths() -> list[str]:
    image_roots = [
        STATIC_ROOT / "tutorials",
        STATIC_ROOT / "examples",
    ]

    pngs: list[str] = []
    seen: set[str] = set()

    for root in image_roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.png"), key=lambda p: p.as_posix().lower()):
            if not path.is_file():
                continue
            rel = path.relative_to(STATIC_ROOT).as_posix().replace("\\", "/")
            url = f"_static/{rel}"
            if url not in seen:
                seen.add(url)
                pngs.append(url)

    return pngs


def main() -> None:
    pngs = collect_png_paths()
    payload = "window.balticRotatorImages = " + repr(pngs) + ";\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(pngs)} homepage rotator images to {OUTPUT}")


if __name__ == "__main__":
    main()
