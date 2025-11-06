#!/usr/bin/env python3
"""
org -> Obsidian Markdown (fixed)

- preserves ATX headings (# Heading 1)
- restores comment blocks (%% ... %%) conservatively (avoids turning headings into comments)
- converts [[id:UUID][Label]] (or pandoc's [Label](id:UUID)) back to [[Label]]
- converts [[https://...][Label]] -> [Label](https://...) (escapes | in label)
- converts file: attachments into ![[name]]
- merges a simple single-line nested list item into its parent line:
    - parent
      - child
  becomes
    - parent - child
"""

from __future__ import annotations
import argparse
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LINK_RE = re.compile(r"\[\[([^\]\[]+?)(?:\]\[([^\]]*?)\])?\]\]")
MD_LINK_WITH_ID_RE = re.compile(r"\[([^\]]+)\]\((id:[^\)]+)\)")
PROPS_RE = re.compile(r"(?s)^:PROPERTIES:\n:ID: .*?\n:END:\n")
TITLE_RE = re.compile(r"(?m)^#\+title:.*\n?")
# matches a contiguous block of lines that start with '# '
COMMENT_BLOCK_RE = re.compile(r"(?:^(?:# .*(?:\n# .*)*)\n?)", flags=re.MULTILINE)
LIST_INDENT_RE = re.compile(r"^([ \t]+)- ", flags=re.MULTILINE)
DOUBLE_QUOTE_BQ_RE = re.compile(r'(?m)^> "(.*)"$')
ATTACH_PREFIXES = ("../attachments/", "attachments/")


def run_pandoc_org_to_md(org_path: Path, md_path: Path):
    subprocess.run(
        [
            "pandoc",
            "--from=org",
            "--to=gfm",
            "--wrap=preserve",
            "-o",
            str(md_path),
            str(org_path),
        ],
        check=True,
    )


def escape_pipe(s: str) -> str:
    return s.replace("|", r"\|")


def link_repl_orgstyle(m: re.Match) -> str:
    """
    Replace org-style [[target][label]] constructs that may survive pandoc (or other org style).
    Handles:
      - full URLs -> [label](url)
      - id:UUID -> [[label]]
      - file: attachments -> ![[filename.ext]] (preserves extension)
      - org files -> [[stem]] or [[stem|label]]
      - other files -> markdown link if label exists, else [[filename.ext]]
    """
    target = m.group(1)
    label = m.group(2)

    # full URL
    if re.match(r"https?://", target):
        lbl = label if label is not None else target
        return f"[{lbl}]({target})"

    # id: -> wiki link
    if target.startswith("id:"):
        return f"[[{label}]]" if label else f"[[{target}]]"

    # file:...
    if target.startswith("file:"):
        raw = target[len("file:") :]
        p = Path(raw)
        # attachments -> embed (keep extension)
        if any(raw.startswith(pref) or f"/{pref}" in raw for pref in ATTACH_PREFIXES):
            return f"![[{p.name}]]"  # <-- keep extension
        # org files -> wiki link with alias if label differs
        if p.suffix.lower() == ".org":
            stem = p.stem
            return f"[[{stem}|{label}]]" if label and label != stem else f"[[{stem}]]"
        # other files -> markdown link if label exists, else wiki link
        return f"[{label}]({raw})" if label else f"[[{p.name}]]"

    # fallback: treat as wiki page
    stem = Path(target).stem
    return f"[[{stem}|{label}]]" if label and label != stem else f"[[{stem}]]"


def md_link_id_to_wiki(md: str) -> str:
    """
    Turn markdown links like [Label](id:UUID) -> [[Label]]
    """

    def _repl(m: re.Match) -> str:
        label = m.group(1)
        return f"[[{label}]]"

    return MD_LINK_WITH_ID_RE.sub(_repl, md)


def convert_comments_conservative(md_text: str) -> str:
    """
    Convert contiguous blocks of '# ' lines into %% comment blocks, but only if
    none of the lines in the block look like an ATX heading (e.g. '# Heading').
    This prevents headings from being converted to comments.
    """

    def _maybe_convert(m: re.Match) -> str:
        block = m.group(0).rstrip("\n")
        lines = block.splitlines()
        # if any line looks like an ATX heading, skip conversion
        heading_pat = re.compile(r"^#{1,6} ")
        if any(heading_pat.match(ln) for ln in lines):
            return block + "\n"
        # strip leading "# " from each line
        stripped = "\n".join(ln[2:] if ln.startswith("# ") else ln for ln in lines)
        return f"%%\n{stripped}\n%%\n"

    return COMMENT_BLOCK_RE.sub(_maybe_convert, md_text)


def fix_list_indent_reverse(md_text: str) -> str:
    """
    Restore list indentation by converting leading spaces before "- " into
    tab-based indent levels. We compute the indentation level as
      level = nspaces // 4
    and emit `\t` repeated `level` times. If there are fewer than 4 spaces,
    we treat it as a single level to avoid producing tiny (1 or 2) space indents.
    """

    def repl(m: re.Match) -> str:
        orig = m.group(1)
        # expand tabs to 4 columns to get a consistent space count
        nspaces = len(orig.expandtabs(4))
        if nspaces == 0:
            # should not match, but keep safe fallback
            return "- "
        # compute level: 4 spaces per level (floor). If <4 spaces but >0, treat as 1 level.
        level = nspaces // 4
        if level == 0:
            level = 1
        return ("\t" * level) + "- "

    return LIST_INDENT_RE.sub(repl, md_text)


def fix_double_blockquotes_reverse(md_text: str) -> str:
    return DOUBLE_QUOTE_BQ_RE.sub(r"> > \1", md_text)


_SUB_SUP_RE = re.compile(r"<(?:sub|sup)>(.*?)</(?:sub|sup)>", flags=re.IGNORECASE)


def fix_subsup_in_link_labels(md_text: str) -> str:
    """
    Robustly replace <sub>...</sub> and <sup>...</sup> inside markdown link labels
    with underscores, then try to repair mangled labels by using the URL's last path
    component (slug) when it contains underscores.

    This handles cases like:
      [scroll<sub>toindex</sub> \| Flutter package](https://.../scroll_to_index)
    -> [scroll_to_index \| Flutter package](https://.../scroll_to_index)
    """
    LINK_LABEL_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def _replace_subs(label: str) -> str:
        # replacement that inserts underscores when adjacent chars are alnum
        out_parts = []
        last_idx = 0
        for m in _SUB_SUP_RE.finditer(label):
            start, end = m.span()
            content = m.group(1)

            before = label[last_idx:start]
            after = label[end:]

            char_before = before[-1] if before else ""
            char_after = after[0] if after else ""

            add_left = bool(
                re.match(r"[A-Za-z0-9]", char_before) and char_before != "_"
            )
            add_right = bool(re.match(r"[A-Za-z0-9]", char_after) and char_after != "_")

            rep = content
            if add_left and not before.endswith("_"):
                rep = "_" + rep
            if add_right and not after.startswith("_"):
                rep = rep + "_"

            out_parts.append(before)
            out_parts.append(rep)
            last_idx = end

        out_parts.append(label[last_idx:])
        new_label = "".join(out_parts)

        # tidy: collapse runs of 3+ underscores
        new_label = re.sub(r"_{3,}", "__", new_label)

        # fix accidental double-escaped pipe sequences (\\| -> \|)
        new_label = new_label.replace(r"\\|", r"\|")

        return new_label

    def _apply_slug_repair(label: str, url: str) -> str:
        """
        If the URL's final path component (slug) contains underscores and the label
        contains the same characters but with underscores missing/shifted, replace
        that token in the label with the canonical slug from the URL.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path or ""
            slug = path.rstrip("/").split("/")[-1]
        except Exception:
            slug = ""

        if not slug or "_" not in slug:
            return label

        slug_no_ = slug.replace("_", "").lower()

        # find the longest alnum/underscore run in the label which, when you remove
        # underscores and lower-case it, equals slug_no_. Replace that run with slug.
        token_re = re.compile(r"[A-Za-z0-9_]+")
        for m in sorted(token_re.finditer(label), key=lambda x: -len(x.group(0))):
            tok = m.group(0)
            if tok.replace("_", "").lower() == slug_no_:
                # replace only this occurrence
                start, end = m.span()
                new_label = label[:start] + slug + label[end:]
                return new_label

        # fallback: if label contains slug_no_ as substring (no underscores), replace
        idx = label.lower().find(slug_no_)
        if idx != -1:
            before = label[:idx]
            after = label[idx + len(slug_no_) :]
            return before + slug + after

        return label

    def repl(m: re.Match) -> str:
        label = m.group(1)
        url = m.group(2)
        new_label = _replace_subs(label)
        new_label = _apply_slug_repair(new_label, url)
        return f"[{new_label}]({url})"

    return LINK_LABEL_RE.sub(repl, md_text)


def strip_added_properties(md_text: str) -> str:
    md_text = PROPS_RE.sub("", md_text)
    md_text = TITLE_RE.sub("", md_text)
    return md_text


def merge_simple_single_child_list(md_text: str) -> str:
    """
    Merge a pattern like:
    - parent
        - child
    into:
    - parent - child

    But skip merging when the parent looks like a category container (ends with ':'
    or contains inline code/backticks), and only merge when the child has exactly one
    extra indentation level. This avoids flattening deeper nested lists or container headers.
    """
    # capture parent line (group1), child's indent (group2), child's content (group3)
    pattern = re.compile(
        r"^([ \t]*-\s+([^\n\S]*)([^\n\S]*[^\n\S]*)?([^\n]+?))\n"  # parent line whole capture
        r"([ \t]{2,})(-\s+([^\n]+))",  # child: leading spaces + "- child"
        flags=re.MULTILINE,
    )

    # Simpler targeted approach: find parent line and the immediately following child
    # but only perform merge when parent doesn't end with ':' and doesn't include inline code
    simple_pattern = re.compile(
        r"^([ \t]*-\s+[^\n]+?)\n([ \t]{2,}-\s+([^\n]+))",
        flags=re.MULTILINE,
    )

    def should_merge(parent: str, child_leading: str) -> bool:
        # parent trimmed text
        ptrim = parent.strip()
        # do not merge if parent is a container header (ends with ':')
        if ptrim.endswith(":"):
            return False
        # do not merge if parent contains inline code/backticks (from =...= / `...`)
        if "`" in parent or "=" in parent:
            # =..= may be left over; be conservative and avoid merging
            return False
        # compute indent levels: count leading tabs as 4 spaces for consistency
        parent_indent = len(re.match(r"^([ \t]*)", parent).group(1).expandtabs(4))
        child_indent = len(re.match(r"^([ \t]*)", child_leading).group(1).expandtabs(4))
        # only merge if child is exactly one level deeper (4 spaces) than parent
        return (child_indent - parent_indent) <= 8 and (
            child_indent - parent_indent
        ) >= 1

    def repl(m: re.Match) -> str:
        parent = m.group(1)
        child_full = m.group(2)
        child = m.group(3)
        # the child_leading we can read from the full match (line beginning)
        # decide whether to merge
        if should_merge(parent, child_full):
            # merge onto same line: "parent - child"
            return f"{parent} - {child}"
        # otherwise keep as-is
        return f"{parent}\n{child_full}"

    # Run replacement repeatedly until it stabilizes
    prev = None
    out = md_text
    while prev != out:
        prev = out
        out = simple_pattern.sub(repl, out)
    return out


LINK_LABEL_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def unescape_pipe_in_link_labels(md_text: str) -> str:
    """
    Turn escaped pipes inside markdown link *labels* (e.g. '\|' or '\\|') back into '|'.
    Only changes the label portion of [label](url).
    """

    def repl(m: re.Match) -> str:
        label = m.group(1).replace(r"\\|", "|").replace(r"\|", "|")
        url = m.group(2)
        return f"[{label}]({url})"

    return LINK_LABEL_RE.sub(repl, md_text)


LIST_ITEM_LINE_RE = re.compile(r"^([ \t]*)([-+*])\s+(.*)$", flags=re.MULTILINE)


def normalize_list_indents_by_column(md_text: str, spaces_per_level: int = 4) -> str:
    """
    Normalize nested list indentation to tabs based on hierarchy.
    Each level = spaces_per_level spaces.
    Preserves deeper nesting correctly.
    """
    out_lines = []
    stack: list[int] = []  # stack of seen indent columns

    for raw_line in md_text.splitlines():
        m = LIST_ITEM_LINE_RE.match(raw_line)
        if not m:
            out_lines.append(raw_line)
            continue

        raw_indent, marker, content = m.group(1), m.group(2), m.group(3)
        col = len(raw_indent.expandtabs(spaces_per_level))

        # Determine level based on stack
        while stack and col < stack[-1]:
            stack.pop()
        if not stack or col > stack[-1]:
            stack.append(col)
        level = len(stack) - 1

        out_lines.append(("\t" * level) + "- " + content)

    return "\n".join(out_lines) + ("\n" if md_text.endswith("\n") else "")


def convert_pandoc_image_links_to_obsidian(md_text: str) -> str:
    """
    Convert Markdown image links like:
      ![](../attachments/foo.png)
    into Obsidian embeds:
      ![[foo.png]]
    """
    IMAGE_RE = re.compile(r"!\[\]\(([^)]+)\)")

    def repl(m: re.Match) -> str:
        path = m.group(1)
        name = Path(path).name  # keep the extension
        return f"![[{name}]]"

    return IMAGE_RE.sub(repl, md_text)


def postprocess_md(md_text: str) -> str:
    md_text = strip_added_properties(md_text)

    # 1) Convert leftover org [[...][...]] constructs and id: links first
    md_text = LINK_RE.sub(link_repl_orgstyle, md_text)

    # Convert Pandoc image links to Obsidian embeds
    md_text = convert_pandoc_image_links_to_obsidian(md_text)

    md_text = md_link_id_to_wiki(md_text)

    # 2) Fix link-label issues (subs, slugs) and unescape pipes in labels
    md_text = fix_subsup_in_link_labels(md_text)
    md_text = unescape_pipe_in_link_labels(md_text)

    # 3) Convert comment blocks conservatively
    md_text = convert_comments_conservative(md_text)

    # 4) NORMALIZE LIST INDENTS ONE TIME (deterministic)
    #    This must be the only list-normalizing pass.
    md_text = normalize_list_indents_by_column(md_text)

    # 5) Restore double blockquote style and final tidy
    md_text = fix_double_blockquotes_reverse(md_text)
    md_text = re.sub(r"[ \t]+$", "", md_text, flags=re.MULTILINE)

    return md_text


def process_file(org_path: Path, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_pandoc_org_to_md(org_path, out_path)
    text = out_path.read_text(encoding="utf-8")
    text = postprocess_md(text)
    out_path.write_text(text, encoding="utf-8")
    return org_path.name


def worker(args):
    return process_file(*args)


def walk_directory(path: Path):
    for p in path.iterdir():
        if p.is_dir():
            yield from walk_directory(p)
        else:
            yield p.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("org_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    in_dir = args.org_directory.resolve()
    out_dir = args.output_directory
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for path in walk_directory(in_dir):
        if path.suffix != ".org":
            tgt = out_dir / path.relative_to(in_dir)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, tgt)
            continue
        rel = path.relative_to(in_dir).with_suffix(".md")
        tgt = out_dir / rel
        jobs.append((path, tgt))

    with ProcessPoolExecutor() as exe:
        for fname in exe.map(worker, jobs):
            print(f"Converted {fname}")


if __name__ == "__main__":
    main()
