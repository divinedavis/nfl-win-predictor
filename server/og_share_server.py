"""Open Graph tags for a shared picks link.

iMessage, Slack and the rest read meta tags out of the HTML and never run the
page's JavaScript, so a link that says whose picks it is has to be rendered by
a server. This one serves /s/<share_id>: it asks Supabase who that card
belongs to, rewrites the <!--OG--> block in the deployed page, and returns the
real app — no redirect, so the URL a person shares is the URL they land on.

Stdlib only, same shape as the hg-og service on the divinedavis droplet.

    systemd: sputter-og.service  ->  /opt/sputter-og/server.py  (127.0.0.1:8792)
    nginx:   location ^~ /s/     ->  proxy_pass, falling back to the static page
"""

import html
import json
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8792
PAGE = "/var/www/nfl/index.html"
SITE = "https://sputterbets.com"
SUPABASE = "https://chhmerckwgytnjhgdiym.supabase.co"
# Publishable key: public by design, and shared_card() is a SECURITY DEFINER
# function that returns aggregates only.
ANON = "sb_publishable_YY0S0mf2V74kwzjNhG_kvA_WzaGTWES"

SHARE_RE = re.compile(r"^/s/([A-Za-z0-9_-]{6,32})/?$")
OG_RE = re.compile(r"<!--OG-->.*?<!--/OG-->", re.S)
TIMEOUT = 4

_page = {"mtime": None, "text": ""}


def page_html() -> str:
    """The deployed app, re-read whenever refresh.sh republishes it."""
    import os
    st = os.stat(PAGE)
    if _page["mtime"] != st.st_mtime:
        with open(PAGE, encoding="utf-8") as fh:
            _page["text"] = fh.read()
        _page["mtime"] = st.st_mtime
    return _page["text"]


def card(share_id: str):
    req = urllib.request.Request(
        f"{SUPABASE}/rest/v1/rpc/shared_card",
        data=json.dumps({"share": share_id}).encode(),
        headers={"apikey": ANON, "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        rows = json.loads(r.read().decode())
    return rows[0] if rows else None


def possessive(name: str) -> str:
    name = name.strip()
    return name + ("'" if name.endswith("s") else "'s")


def describe(c: dict) -> tuple:
    picks = c.get("picks") or []
    weeks = [p["week"] for p in picks if p.get("week") is not None]
    who = possessive(c.get("display_name") or "Someone")
    title = f"{who} Picks for Week {max(weeks)}" if weeks else f"{who} Picks"

    wins, losses = c.get("wins") or 0, c.get("losses") or 0
    n = len(picks)
    parts = [f"{n} pick" + ("" if n == 1 else "s")]
    if wins + losses:
        pct = c.get("pct")
        parts.append(f"{wins}–{losses}" +
                     (f" ({float(pct):.0f}%)" if pct is not None else ""))
    else:
        parts.append("nothing settled yet")
    parts.append("game winners and player over/unders on Sputter Bets")
    return title, " · ".join(parts)


def og_block(title: str, description: str, url: str) -> str:
    t, d = html.escape(title, quote=True), html.escape(description, quote=True)
    return (
        "<!--OG-->\n"
        f"<title>{html.escape(title)}</title>\n"
        '<meta property="og:type" content="website">\n'
        '<meta property="og:site_name" content="Sputter Bets">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'<meta property="og:url" content="{html.escape(url, quote=True)}">\n'
        f'<meta property="og:image" content="{SITE}/og.jpg">\n'
        '<meta property="og:image:width" content="960">\n'
        '<meta property="og:image:height" content="600">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{t}">\n'
        f'<meta name="twitter:description" content="{d}">\n'
        f'<meta name="twitter:image" content="{SITE}/og.jpg">\n'
        "<!--/OG-->")


class Handler(BaseHTTPRequestHandler):
    server_version = "sputter-og"

    def log_message(self, fmt, *args):        # journald already timestamps
        pass

    def _send(self, body: str, status: int = 200):
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        m = SHARE_RE.match(self.path.split("?")[0])
        if not m:
            # Anything else under /s/ is not a card; let nginx serve the app.
            self.send_error(404)
            return
        share_id = m.group(1)
        try:
            page = page_html()
        except OSError:
            self.send_error(503)
            return
        try:
            c = card(share_id)
        except (urllib.error.URLError, ValueError, TimeoutError):
            c = None                    # Supabase unreachable: generic preview
        if c:
            title, desc = describe(c)
        else:
            title = "Sputter Bets — a shared card"
            desc = ("Somebody's game and player-prop picks. Nothing is staked "
                    "on any of it.")
        block = og_block(title, desc, f"{SITE}/s/{share_id}")
        self._send(OG_RE.sub(lambda _: block, page, count=1))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
