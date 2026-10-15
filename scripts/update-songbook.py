#!/usr/bin/env python3
"""Refresh the static songbook after editing or adding assets/lyrics/*.txt."""
from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "jellohead/songs/index.html"


def render_songs():
    entries = []
    files = sorted((ROOT / "assets/lyrics").glob("*.txt"), key=lambda p: p.stem.lower())
    for number, path in enumerate(files, 1):
        title = path.stem.replace("_", " ")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        lyrics = escape(path.read_text().strip())
        # Preserve source whitespace without trailing whitespace in the HTML.
        lyrics = re.sub(r"[ \t]+$", lambda match: "".join(f"&#{ord(char)};" for char in match[0]), lyrics, flags=re.MULTILINE)
        entries.append(f'''      <details class="song" id="{slug}" name="songs">
        <summary><span class="song-number" aria-hidden="true">{number:02}</span><h2 class="song-title">{escape(title)}</h2></summary>
        <div class="song-body">
          <pre class="lyrics">{lyrics}</pre>
          <div class="song-actions">
            <a href="#{slug}">Link to this song</a>
            <a href="/assets/lyrics/{escape(path.name, quote=True)}" download>Download lyrics</a>
          </div>
        </div>
      </details>''')
    return "\n".join(entries)


if __name__ == "__main__":
    page = PAGE.read_text()
    before, rest = page.split("<!-- SONGS START -->", 1)
    _, after = rest.split("<!-- SONGS END -->", 1)
    PAGE.write_text(before + "<!-- SONGS START -->\n" + render_songs() + "\n<!-- SONGS END -->" + after)
    print("Updated songbook from assets/lyrics/*.txt")
