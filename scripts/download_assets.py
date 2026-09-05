"""Populate the GRAFT data root.

    python scripts/download_assets.py                # everything needed to run
    python scripts/download_assets.py --eval         # + benchmark assets
    python scripts/download_assets.py --check        # report what is present
    python scripts/download_assets.py --skip-gated   # skip anything needing an account

Three kinds of asset, all fetched by this one script:

* **Ours** — weights and the benchmark assets we produced, from the Hugging Face
  Hub (`ympradyumna/graft`).
* **Public** — the NLF detector, a direct download from its GitHub release.
* **Account-gated** — the SMPL-X body models, from the MPI download portal. You
  are prompted for the credentials of the site that hosts them; register first at
  https://smpl-x.is.tue.mpg.de.

Credentials are used for the download and not stored.

The data root is `$GRAFT_DATA_ROOT` (default `./data`). `$GRAFT_HF_REPO` points the
Hub source elsewhere; `$HF_TOKEN` authenticates it if it is private.
"""
from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import shutil
import subprocess
import tempfile
import sys
import zipfile
from pathlib import Path
import string

#: Percent-encoding identical to the `urle()` helper in the MPI download scripts
#: (e.g. Phy-SIC's fetch_data.sh): everything except [A-Za-z0-9.~-] is escaped.
#: Python's urllib quote() leaves "_" and "/" alone, which this does not.
_UNRESERVED = frozenset(string.ascii_letters + string.digits + ".~-")


def urle(s: str) -> str:
    return "".join(
        c if c in _UNRESERVED else "".join(f"%{b:02X}" for b in c.encode())
        for c in s)

MANIFEST = Path(__file__).with_name("assets.json")
MPI_DOWNLOAD = "https://download.is.tue.mpg.de/download.php"

#: Where to register for each gated domain.
REGISTER_AT = {"smplx": "https://smpl-x.is.tue.mpg.de", "prox": "https://prox.is.tue.mpg.de",
               "agora": "https://agora.is.tue.mpg.de"}


def data_root() -> Path:
    return Path(os.environ.get("GRAFT_DATA_ROOT", "data")).expanduser().resolve()


def satisfied(root: Path, entry: dict):
    """The path satisfying this entry, or None. Accepts the listed alternatives, so
    e.g. either NLF build counts."""
    for rel in [entry["path"], *entry.get("alt_paths", [])]:
        p = root / rel
        if (p.is_dir() and any(p.iterdir())) or (p.is_file()):
            return rel
    return None


# ---------------------------------------------------------------------------
# Account-gated downloads (MPI portal)
# ---------------------------------------------------------------------------

class Portal:
    """Prompts once per site, then downloads with the credentials kept off argv."""

    def __init__(self):
        self._creds: dict[str, str] = {}

    def _body(self, domain: str) -> str:
        if domain not in self._creds:
            where = REGISTER_AT.get(domain, f"https://{domain}.is.tue.mpg.de")
            print(f"\n  {domain}: register at {where}, then enter those credentials.")
            user = os.environ.get(f"{domain.upper()}_USERNAME") or input("    username (email): ").strip()
            pw = os.environ.get(f"{domain.upper()}_PASSWORD") or getpass.getpass("    password: ")
            self._creds[domain] = f"username={urle(user)}&password={urle(pw)}"
        return self._creds[domain]

    def download(self, domain: str, sfile: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # wget, not curl, and this is load-bearing. The portal answers the POST
        # login with `Set-Cookie: PHPSESSID=...` and a 302; the redirect target
        # needs that cookie. wget carries cookies across redirects within one
        # invocation. curl sends none unless a cookie engine is enabled (-b/-c),
        # so `curl -L --data ...` follows the redirect with no session and is
        # handed the sign-in page instead of the archive -- HTTP 200, HTML on
        # disk, and a "wrong password" that is nothing of the sort.
        #
        # This is the invocation the MPI download scripts use (e.g. Phy-SIC's
        # fetch_data.sh), unchanged. The portal also rate-limits by IP and will
        # start returning 403 to everything after a burst of attempts, so a
        # failed attempt is expensive. Verified working end to end; leave it be.
        #
        #   wget --post-data "username=$u&password=$p" \
        #        'https://download.is.tue.mpg.de/download.php?domain=X&sfile=Y' \
        #        -O dest --no-check-certificate --continue
        #
        # The body goes via --post-file rather than --post-data only so the
        # password stays out of argv; the bytes on the wire are identical.
        # (`resume=1` is absent because the server appends it to its own redirect
        # target; sending it ourselves is neither needed nor harmful.)
        url = f"{MPI_DOWNLOAD}?domain={domain}&sfile={sfile}"
        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            os.chmod(fh.name, 0o600)
            fh.write(self._body(domain))
            body_file = fh.name
        try:
            cmd = ["wget", f"--post-file={body_file}", url, "-O", str(dest),
                   "--no-check-certificate", "--continue"]
            r = subprocess.run(cmd)
        finally:
            os.unlink(body_file)
        if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"{domain}/{sfile}: wget exited {r.returncode}")

        # The portal answers a rejected login with its sign-in page, HTTP 200. Without
        # this check that HTML lands on disk named like the archive and fails later as
        # a confusing "not a zip file".
        head = dest.open("rb").read(64)
        if head[:2] != b"PK":
            dest.unlink(missing_ok=True)
            where = REGISTER_AT.get(domain, f"https://{domain}.is.tue.mpg.de")
            if head.lstrip()[:14].lower().startswith(b"<!doctype html") or head[:5].lower() == b"<html":
                raise RuntimeError(
                    f"{domain} rejected the login (it returned its sign-in page).\n"
                    f"    Each MPI project site is a separate registration, even with the "
                    f"same email.\n"
                    f"    Register and accept the licence at {where}, then retry.")
            raise RuntimeError(f"{domain}/{sfile}: downloaded data is not a zip "
                               f"(starts with {head[:8]!r})")


def _find_in_zip(z: zipfile.ZipFile, member: str):
    """Locate a member by exact path, by trailing path, or by basename."""
    want = member.split("/")[-1].lower()
    names = z.namelist()
    for n in names:
        if n == member or n.endswith("/" + member):
            return n
    for n in names:
        if n.split("/")[-1].lower() == want:
            return n
    return None


def _extract_one(z: zipfile.ZipFile, name: str, dest: Path) -> None:
    """Write one zip member out, and refuse to leave a short file behind.

    A truncated extraction is worse than a failed one: every later check --
    ``satisfied()``, ``check_install.py`` -- only asks whether the path exists, so
    the run reports "Data root is complete" and the corruption first shows up as a
    BadZipFile from deep inside model loading.
    """
    want = z.getinfo(name).file_size
    with z.open(name) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    got = dest.stat().st_size
    if got != want:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: extracted {got:,} bytes but the archive says "
            f"{want:,}. The partial file has been removed; re-run to retry.")


def extract_member(archive: Path, member: str, dest: Path) -> None:
    """Pull one file out of a zip.

    Matches on the exact path, the trailing path, or just the basename, so a
    differing top-level directory does not break it. If the archive nests another
    zip (SMPL-X ships its .npz models this way), that one is searched too.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        hit = _find_in_zip(z, member)
        if hit is not None:
            _extract_one(z, hit, dest)
            return
        for nested in [n for n in z.namelist() if n.lower().endswith(".zip")]:
            with z.open(nested) as raw:
                data = io.BytesIO(raw.read())
            with zipfile.ZipFile(data) as z2:
                inner = _find_in_zip(z2, member)
                if inner is not None:
                    _extract_one(z2, inner, dest)
                    return
        names = z.namelist()
        want = member.split("/")[-1]
        stem = want.rsplit(".", 1)[0].lower()
        near = [n for n in names if stem in n.lower()]
        hint = (f" The archive does contain {near[:3]}, so this is probably a naming "
                f"or format difference." if near else "")
        raise RuntimeError(
            f"{want} not found in {archive.name}. It holds {len(names)} entries, "
            f"e.g. {names[:4]}.{hint} Update the 'member' field in "
            f"scripts/assets.json to match.")


# ---------------------------------------------------------------------------
# Public direct download
# ---------------------------------------------------------------------------

def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-L", "--fail", "--progress-bar",
                        "--continue-at", "-", "-o", str(dest), url])
    if r.returncode != 0:
        raise RuntimeError(f"{url} failed (curl exit {r.returncode})")


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

def synthesize(kind: str, dest: Path, root: Path) -> None:
    """Produce an asset locally instead of downloading it.

    ``kid_template``: smplfitter loads ``kid_template.npy`` for every smpl-family
    model and turns it into ``kid_shapedir``, which enters the mesh only as
    ``einsum('vc,b->bvc', kid_shapedir, kid_factor)``. ``kid_factor`` defaults to
    zero and GRAFT never sets it, so the file's contents cannot affect any output
    -- verified by running the body model against a real template, a zero one and
    a deliberately corrupted one: bit-identical vertices in all three cases.

    The real template is gated behind a separate AGORA registration. Since it is
    provably inert here, we write zeros of the right shape rather than make every
    user register at another site for a file that does nothing.
    """
    import numpy as np
    if kind != "kid_template":
        raise RuntimeError(f"unknown synthesised asset: {kind}")
    neutral = root / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not neutral.exists():
        raise RuntimeError(f"need {neutral} first to size the kid template")
    with np.load(neutral, allow_pickle=True) as z:
        n = len(z["v_template"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, np.zeros((n, 3), dtype=np.float64))


def fetch_repo_file(repo: str, repo_file: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=repo_file, repo_type="model")
    # Copy rather than link: the data root should survive a HF cache purge.
    shutil.copyfile(cached, dest)


def fetch_repo_dir(repo: str, repo_dir: str, dest: Path) -> int:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=repo, repo_type="model", allow_patterns=f"{repo_dir}/*")
    n = 0
    for f in sorted((Path(local) / repo_dir).iterdir()):
        if f.is_file():
            shutil.copyfile(f, dest / f.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="Also fetch the benchmark assets.")
    ap.add_argument("--check", action="store_true", help="Report only; download nothing.")
    ap.add_argument("--skip-gated", action="store_true",
                    help="Skip anything needing an account, and say what to fetch by hand.")
    ap.add_argument("--data-root", default=None, help="Overrides $GRAFT_DATA_ROOT.")
    args = ap.parse_args()

    if args.data_root:
        os.environ["GRAFT_DATA_ROOT"] = args.data_root
    root = data_root()
    spec = json.loads(MANIFEST.read_text())
    repo = os.environ.get("GRAFT_HF_REPO", spec["hf_repo"])
    print(f"data root : {root}")
    print(f"hub repo  : {repo}")

    portal, archives, skipped, failed = Portal(), {}, [], []
    for f in spec["files"]:
        if f.get("eval") and not (args.eval or args.check):
            continue
        dest = root / f["path"]
        found = satisfied(root, f)
        if found:
            note = "" if found == f["path"] else "   (accepted alternative)"
            print(f"  have    {found}{note}")
            continue
        if args.check:
            print(f"  missing {f['path']}")
            failed.append(f["path"])
            continue
        if f.get("gated") and args.skip_gated:
            print(f"  skipped {f['path']}  (needs a {f['gated']} account)")
            skipped.append(f)
            continue
        try:
            if "repo_dir" in f:
                n = fetch_repo_dir(repo, f["repo_dir"], dest)
                print(f"  got     {f['path']}  ({n} files)")
            elif "repo_file" in f:
                fetch_repo_file(repo, f["repo_file"], dest)
                print(f"  got     {f['path']}  ({dest.stat().st_size/1e6:.1f} MB)")
            elif "url" in f:
                print(f"  get     {f['path']}")
                download_url(f["url"], dest)
            elif "synthesize" in f:
                print(f"  make    {f['path']}")
                synthesize(f["synthesize"], dest, root)
                print(f"  made    {f['path']}  ({dest.stat().st_size/1e6:.2f} MB)")
            elif "gated" in f:
                domain = f["gated"]
                if "archive" not in f:
                    # Some gated assets are single files, not zip members.
                    print(f"  get     {f['path']}  ({domain} account)")
                    portal.download(domain, f["sfile"], dest)
                    print(f"  got     {f['path']}  ({dest.stat().st_size/1e6:.1f} MB)")
                    continue
                name = f["archive"]
                key = (domain, name)
                if key not in archives:
                    tmp = root / "_downloads" / name
                    if not tmp.exists():
                        print(f"  get     {name}  (for {f['path']})")
                        portal.download(domain, name, tmp)
                    archives[key] = tmp
                extract_member(archives[key], f["member"], dest)
                print(f"  got     {f['path']}  ({dest.stat().st_size/1e6:.1f} MB)")
            else:
                raise RuntimeError("manifest entry has no source")
        except Exception as e:
            print(f"  FAILED  {f['path']}: {type(e).__name__}: {e}")
            failed.append(f["path"])

    if archives and not args.check:
        keep = root / "_downloads"
        print(f"\n  (archives kept in {keep}; delete it to reclaim the space)")

    if skipped:
        print("\nSkipped, fetch these yourself:\n")
        for f in skipped:
            print(f"  {root / f['path']}")
            print(f"    {f.get('note', '')}\n")

    if failed:
        print(f"\n{len(failed)} item(s) missing or failed.")

    ok = not failed and not skipped
    print("\nData root is complete." if ok else "\nData root is INCOMPLETE.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
