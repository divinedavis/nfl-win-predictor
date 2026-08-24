"""Re-inject the live data blob into a freshly scp'd template and publish it.

Front-end-only changes do not need a retrain. `refresh.sh` rebuilds
web/index.html from web/template.html through export_web.py, which pulls
nflverse data and runs the model — minutes of work to change a button. This
takes the JSON payload already sitting in the deployed index.html, drops it
into the new template, and writes the page back.

    scp web/template.html root@104.248.12.129:/opt/nfl-predictor/web/
    ssh root@104.248.12.129 'cd /opt/nfl-predictor && python3 scripts/redeploy_web.py'

Note that /opt/nfl-predictor on the droplet is a plain copy, not a git
checkout — `git pull` there fails, so the template has to be scp'd.
"""

import re
import shutil
from pathlib import Path

DOCROOT = Path("/var/www/nfl")


def main() -> None:
    idx = Path("web/index.html")
    blob = re.search(r"^  const DATA = (.*);$", idx.read_text(), re.M)
    if not blob:
        raise SystemExit("no `const DATA = ...;` line in web/index.html")
    out = Path("web/template.html").read_text().replace("__DATA_JSON__", blob.group(1))
    if "__DATA_JSON__" in out:
        raise SystemExit("template placeholder left unfilled")
    idx.write_text(out)
    print(f"rebuilt web/index.html ({len(out)} bytes)")
    if DOCROOT.is_dir():
        shutil.copy(idx, DOCROOT / "index.html")
        shutil.copy("web/how-it-works.html", DOCROOT / "how-it-works.html")
        print(f"deployed to {DOCROOT}")


if __name__ == "__main__":
    main()
