"""Build the BlenderGIS release zips and the extension repository index, and publish them.

Blender can follow a remote extension repository and offer its updates: users add
	https://github.com/mnmly/BlenderGIS/releases/latest/download/index.json
in Preferences > Get Extensions > Repositories. GitHub redirects that URL to the
index.json asset of the latest release, so each release ships its own index.json
listing the zips of that release.

Usage (from the repository root, wheels fetched with tools/fetch_wheels.py):
	uv run tools/release.py                           # build dist/<version>/ (zips + index.json) from the committed HEAD
	uv run tools/release.py --publish --notes notes.md  # also create the GitHub release v<version>
	uv run tools/release.py --index-only --publish    # add index.json to an already published release

After publishing, the repositories in DOWNSTREAM_REPOS are asked to rebuild their index.

The Blender executable is taken from $BLENDER, else /Applications/Blender.app, else `blender`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "blender_manifest.toml"
DIST = ROOT / "dist"
GITHUB_REPO = "mnmly/BlenderGIS"
REPO_URL = f"https://github.com/{GITHUB_REPO}/releases/latest/download/index.json"
# Extension repositories that merge this index into theirs, rebuilt after each publish
DOWNSTREAM_REPOS = ["mnmly/aa-blender-extensions"]


def blenderExecutable():
	if os.environ.get("BLENDER"):
		return os.environ["BLENDER"]
	mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
	if mac.exists():
		return str(mac)
	found = shutil.which("blender")
	if found is None:
		sys.exit("Blender not found, set $BLENDER to its executable")
	return found


def run(cmd, **kwargs):
	print("$", " ".join(str(c) for c in cmd))
	return subprocess.run(cmd, check=True, **kwargs)


def checkCommitted():
	'''The zips are built from HEAD, uncommitted changes to tracked files would be left out'''
	out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
		capture_output=True, text=True, check=True).stdout
	if out.strip():
		sys.exit("Uncommitted changes would be left out of the release, commit them or pass --allow-dirty:\n" + out)


def build(blender, outDir):
	'''Build the zips from a clean export of HEAD plus the (git ignored) wheels: local files are left out'''
	wheels = sorted((ROOT / "wheels").glob("*.whl"))
	if not wheels:
		sys.exit("No wheels in ./wheels, run `uv run --with pip tools/fetch_wheels.py` first")
	if outDir.exists():
		shutil.rmtree(outDir)
	outDir.mkdir(parents=True)
	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "blendergis"
		src.mkdir()
		archive = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
		subprocess.run(["tar", "-x", "-C", src], input=archive, check=True)
		(src / "wheels").mkdir()
		for wheel in wheels:
			shutil.copy2(wheel, src / "wheels")
		run([blender, "--factory-startup", "--command", "extension", "build", "--split-platforms",
			"--source-dir", src, "--output-dir", outDir])


def downloadZips(version, outDir):
	'''Zips of an already published release, to index them'''
	if outDir.exists():
		shutil.rmtree(outDir)
	outDir.mkdir(parents=True)
	run(["gh", "release", "download", f"v{version}", "-R", GITHUB_REPO, "-p", "*.zip", "-D", outDir])


def generateIndex(blender, version, outDir):
	'''index.json of the zips in outDir, their archive urls pointing to the GitHub release assets'''
	zips = sorted(outDir.glob(f"blendergis-{version}-*.zip"))
	if not zips:
		sys.exit(f"No blendergis-{version}-*.zip in {outDir}")
	run([blender, "--factory-startup", "--command", "extension", "server-generate", "--repo-dir", outDir])
	indexPath = outDir / "index.json"
	index = json.loads(indexPath.read_text())
	base = f"https://github.com/{GITHUB_REPO}/releases/download/v{version}/"
	for entry in index["data"]:
		entry["archive_url"] = base + entry["archive_url"].removeprefix("./")
	indexPath.write_text(json.dumps(index, indent=2))
	platforms = sorted(p for entry in index["data"] for p in entry.get("platforms", []))
	print(f"{indexPath}: {len(index['data'])} packages ({', '.join(platforms)})")
	return zips, indexPath


def releaseExists(version):
	return subprocess.run(["gh", "release", "view", f"v{version}", "-R", GITHUB_REPO],
		capture_output=True).returncode == 0


def publish(version, files, notes):
	if releaseExists(version):
		run(["gh", "release", "upload", f"v{version}", "-R", GITHUB_REPO, "--clobber", *files])
		return
	if notes is None:
		sys.exit(f"Release v{version} does not exist yet, pass --notes with its description")
	run(["gh", "release", "create", f"v{version}", "-R", GITHUB_REPO, "--title", f"BlenderGIS {version}",
		"--notes-file", notes, *files])


def notifyDownstream():
	'''Trigger a rebuild of the repositories that merge this index, so they offer the release right away'''
	for repo in DOWNSTREAM_REPOS:
		result = subprocess.run(["gh", "api", "-X", "POST", f"repos/{repo}/dispatches",
			"-f", "event_type=upstream-release"], capture_output=True, text=True)
		if result.returncode == 0:
			print(f"Notified {repo}")
		else:
			print(f"Could not notify {repo} (it rebuilds daily anyway): {result.stderr.strip()}")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--publish", action="store_true", help="upload to the GitHub release v<version> (created if needed)")
	parser.add_argument("--notes", help="markdown file of the release description, to create the release")
	parser.add_argument("--index-only", action="store_true", help="index the zips of the published release instead of building")
	parser.add_argument("--allow-dirty", action="store_true", help="build HEAD even with uncommitted changes")
	args = parser.parse_args()

	version = tomllib.loads(MANIFEST.read_text())["version"]
	outDir = DIST / version
	blender = blenderExecutable()
	if args.index_only:
		downloadZips(version, outDir)
	else:
		if not args.allow_dirty:
			checkCommitted()
		build(blender, outDir)
	zips, indexPath = generateIndex(blender, version, outDir)
	if args.publish:
		publish(version, [indexPath] if args.index_only else [*zips, indexPath], args.notes)
		notifyDownstream()
	print(f"\nUsers get updates from the repository URL:\n  {REPO_URL}")


if __name__ == "__main__":
	main()
