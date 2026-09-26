"""HTML-версии отчётов: на Windows их удобнее открыть в браузере, чем .md."""
from __future__ import annotations

import html
import re

CSS = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d9d9de; --head:#f4f4f6; --accent:#0a66c2; --bad:#c62828; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --fg:#e8e8ea; --muted:#9a9aa2; --line:#33343a; --head:#1f2024; --accent:#6aa8ff; --bad:#ff6b6b; }
}
body { background:var(--bg); color:var(--fg); font:15px/1.5 "Segoe UI", system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }
h1 { font-size:22px; } h2 { font-size:18px; margin-top:28px; }
a { color:var(--accent); }
table { border-collapse:collapse; margin:8px 0 16px; width:100%; }
th, td { border:1px solid var(--line); padding:4px 8px; vertical-align:top; }
th { background:var(--head); text-align:left; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
code { background:var(--head); padding:1px 4px; border-radius:3px; }
img, video { max-width:100%; border-radius:4px; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:8px; }
.gallery figure { margin:0; } .gallery figcaption { color:var(--muted); font-size:13px; }
.wrap { overflow-x:auto; }
"""

_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"), r'<img alt="\1" src="\2">'),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
]
_NUM = re.compile(r"^[\d\s.,%:—-]+$")


def _inline(text: str) -> str:
    s = html.escape(text, quote=False)
    for rx, rep in _INLINE:
        s = rx.sub(rep, s)
    return s


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(md: str) -> str:
    """Подмножество Markdown, которое пишут наши отчёты: заголовки, абзацы, списки, таблицы,
    жирный, код, ссылки, картинки."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            t = ["<div class=wrap><table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
            for r in body:
                t.append("<tr>" + "".join(
                    f"<td class=num>{_inline(c)}</td>" if c and _NUM.match(c) else f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</tbody></table></div>")
        elif re.match(r"^\s*- ", line):
            items = []
            while i < len(lines) and re.match(r"^\s*- ", lines[i]):
                items.append(f"<li>{_inline(lines[i].split('- ', 1)[1])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3}\s|\s*- |\s*\|)", lines[i]):
                para.append(_inline(lines[i]))
                i += 1
            out.append("<p>" + "<br>".join(para) + "</p>")
    return "\n".join(out)


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=ru><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>")
