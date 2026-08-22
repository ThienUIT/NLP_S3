#!/usr/bin/env python3
"""Trich toan bo text tu relation/main/index.html ra content.md (markdown).

Muc tieu: giu NGUYEN VAN moi chu trong slide, khong tom tat, khong dien giai.
"""
import html as htmlmod
import re
import sys
from html.parser import HTMLParser

SRC = sys.argv[1]
DST = sys.argv[2]

SKIP_TAGS = {"script", "style", "head", "title"}
BLOCK = {"div", "section", "p", "li", "ul", "ol", "figure", "figcaption", "table",
         "tr", "h1", "h2", "h3", "h4", "pre", "span"}


class Node:
    def __init__(self, tag, attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.kids = []

    @property
    def cls(self):
        return self.attrs.get("class", "").split()

    def text(self):
        out = []
        for k in self.kids:
            out.append(k if isinstance(k, str) else k.text())
        return "".join(out)


class Tree(HTMLParser):
    VOID = {"br", "img", "hr", "meta", "link", "input"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.root = Node("root")
        self.stack = [self.root]
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        n = Node(tag, attrs)
        self.stack[-1].kids.append(n)
        if tag not in self.VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        if self.skip:
            return
        self.stack[-1].kids.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag in self.VOID:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, d):
        if not self.skip:
            self.stack[-1].kids.append(d)

    def handle_entityref(self, name):
        if not self.skip:
            self.stack[-1].kids.append(htmlmod.unescape("&%s;" % name))

    def handle_charref(self, name):
        if not self.skip:
            self.stack[-1].kids.append(htmlmod.unescape("&#%s;" % name))


def norm(s):
    return re.sub(r"[ \t\n\r]+", " ", s)


def inline(node, in_pre=False):
    """Render inline content -> markdown string, giu nguyen chu."""
    out = []
    for k in node.kids:
        if isinstance(k, str):
            out.append(k if in_pre else norm(k))
            continue
        t, c = k.tag, k.cls
        if t == "br":
            out.append("\n" if in_pre else "  \n")
        elif t == "img":
            out.append("\n![%s](%s)\n" % (k.attrs.get("alt", ""), k.attrs.get("src", "")))
        elif t in ("b", "strong"):
            inner = inline(k, in_pre).strip()
            out.append("**%s**" % inner if inner else "")
        elif t in ("i", "em"):
            out.append("*%s*" % inline(k, in_pre).strip())
        elif t == "code":
            out.append("`%s`" % inline(k, in_pre).strip())
        elif t == "sub":
            out.append("_%s" % inline(k, in_pre).strip())
        elif t == "sup":
            out.append("^%s" % inline(k, in_pre).strip())
        elif t == "a":
            href = k.attrs.get("href", "")
            txt = inline(k, in_pre).strip()
            out.append("[%s](%s)" % (txt, href) if href else txt)
        else:
            out.append(inline(k, in_pre))
    return "".join(out)


def flush(buf, s):
    s = s.rstrip()
    if s:
        buf.append(s)


def render_table(node, buf):
    rows = []
    for tr in iter_tag(node, "tr"):
        cells, hdr = [], False
        for c in tr.kids:
            if isinstance(c, Node) and c.tag in ("td", "th"):
                if c.tag == "th":
                    hdr = True
                cells.append(inline(c).strip().replace("|", "\\|") or " ")
        if cells:
            rows.append((hdr, cells))
    if not rows:
        return
    ncol = max(len(r[1]) for r in rows)
    lines = []
    started = False
    for i, (hdr, cells) in enumerate(rows):
        cells = cells + [" "] * (ncol - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
        if not started and (hdr or i == 0):
            lines.append("|" + "---|" * ncol)
            started = True
    buf.append("\n".join(lines))


def iter_tag(node, tag):
    for k in node.kids:
        if isinstance(k, Node):
            if k.tag == tag:
                yield k
            else:
                yield from iter_tag(k, tag)


def walk(node, buf, depth=0):
    """Duyet cay, xuat markdown theo block."""
    t, c = node.tag, node.cls

    # --- cac block la (leaf), xu ly truc tiep ---
    if t == "table":
        render_table(node, buf)
        return
    if t == "pre":
        flush(buf, "```\n%s\n```" % inline(node, in_pre=True).strip("\n"))
        return
    if t in ("h1", "h2"):
        flush(buf, "### " + inline(node).strip().replace("  \n", " — "))
        return
    if t == "h3":
        flush(buf, "**%s**" % inline(node).strip())
        return
    if t == "h4":
        flush(buf, "#### " + inline(node).strip())
        return
    if t == "figure":
        img = next(iter_tag(node, "img"), None)
        if img is not None:
            buf.append("![%s](%s)" % (img.attrs.get("alt", ""), img.attrs.get("src", "")))
        for fc in iter_tag(node, "figcaption"):
            flush(buf, "*%s*" % inline(fc).strip())
        return
    if t in ("ul", "ol"):
        items = []
        for i, li in enumerate([k for k in node.kids if isinstance(k, Node) and k.tag == "li"], 1):
            mark = "%d." % i if t == "ol" else "-"
            body = inline(li).strip()
            body = body.replace("  \n", "\n    ")
            items.append("%s %s" % (mark, body))
        if items:
            buf.append("\n".join(items))
        return
    if t == "p":
        txt = inline(node).strip()
        if txt:
            buf.append("> " + txt.replace("  \n", "  \n> ") if "note" in node.attrs.get("class", "") else txt)
        return

    # --- div/span co class dac biet ---
    if t in ("div", "span"):
        if "kicker" in c:
            flush(buf, "`%s`" % inline(node).strip())
            return
        if "lbl" in c:
            flush(buf, "**[%s]**" % inline(node).strip())
            return
        if "eq" in c:
            flush(buf, "```\n%s\n```" % inline(node, in_pre=True).strip())
            return
        if "note" in c:
            txt = inline(node).strip().replace("  \n", "\n> ")
            flush(buf, "> " + txt)
            return
        if "chips" in c:
            chips = [inline(k).strip() for k in node.kids
                     if isinstance(k, Node) and "chip" in k.cls]
            if chips:
                flush(buf, " · ".join(
                    c if ("**" in c or "`" in c) else "`%s`" % c for c in chips))
            return
        if "stat" in c:
            n = next((inline(k).strip() for k in node.kids
                      if isinstance(k, Node) and "n" in k.cls), "")
            kk = next((inline(k).strip() for k in node.kids
                       if isinstance(k, Node) and "k" in k.cls), "")
            flush(buf, "- **%s** — %s" % (n, kk.replace("  \n", " ")))
            return
        if "ref" in c and "reflist" not in c:
            flush(buf, "- " + inline(node).strip().replace("  \n", "  \n  "))
            return
        if "cap" in c and not any(isinstance(k, Node) and k.tag in BLOCK - {"span"}
                                  for k in node.kids):
            flush(buf, "*%s*" % inline(node).strip())
            return
        if "sub" in c or "small" in c or "tiny" in c or "authors" in c:
            # co the chua the con -> chi lay text neu khong co block con
            if not any(isinstance(k, Node) and k.tag in BLOCK - {"span"} for k in node.kids):
                flush(buf, inline(node).strip())
                return
        if "foot" in c or "prog" in c:
            return

    # --- container: neu khong co con la block thi in text; nguoc lai de quy ---
    has_block_kid = any(isinstance(k, Node) and k.tag in BLOCK or
                        (isinstance(k, Node) and k.tag in ("table", "pre", "figure", "img"))
                        for k in node.kids)
    if not has_block_kid:
        flush(buf, inline(node).strip())
        return

    # in phan text truc tiep (text node truoc block con) neu co
    lead = "".join(k for k in node.kids if isinstance(k, str))
    if norm(lead).strip():
        flush(buf, norm(lead).strip())

    for k in node.kids:
        if isinstance(k, Node):
            walk(k, buf, depth + 1)


def main():
    src = open(SRC, encoding="utf-8").read()
    tr = Tree()
    tr.feed(src)

    slides = [n for n in iter_tag(tr.root, "section") if "slide" in n.cls]
    out = []
    out.append("# LeWorldModel (LeWM) — Nội dung slide (trích xuất đầy đủ)\n")
    out.append("> File này được **trích xuất tự động** từ [`index.html`](index.html) — "
               "giữ nguyên văn toàn bộ chữ trên slide, dùng để đối chiếu với bài báo gốc "
               "(`../docs/LeWorldModel.pdf`).\n")
    out.append("**Mục lục**\n")
    for i, s in enumerate(slides, 1):
        ti = s.attrs.get("data-title", "?")
        anchor = re.sub(r"[^\w\s-]", "", ti.lower()).strip().replace(" ", "-")
        out.append("%d. [%s](#%d-%s)" % (i, ti, i, anchor))
    out.append("\n---\n")

    for i, s in enumerate(slides, 1):
        ti = s.attrs.get("data-title", "?")
        out.append("## %d. %s\n" % (i, ti))
        # footer label (nguon section trong paper)
        foot = next((n for n in iter_tag(s, "div") if "foot" in n.cls), None)
        buf = []
        for k in s.kids:
            if isinstance(k, Node):
                walk(k, buf)
        out.append("\n\n".join(b for b in buf if b.strip()))
        if foot is not None:
            ftxt = norm(foot.text()).strip()
            if ftxt:
                out.append("\n*<sub>%s</sub>*" % ftxt)
        out.append("\n---\n")

    text = "\n".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    open(DST, "w", encoding="utf-8").write(text)
    print("wrote %s — %d slides, %d chars" % (DST, len(slides), len(text)))


main()
