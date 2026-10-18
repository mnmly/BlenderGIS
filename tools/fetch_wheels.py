"""Download the Python wheels bundled with the BlenderGIS extension.

Blender extensions must not install packages at runtime, so third-party
dependencies ship as wheels listed in blender_manifest.toml. This script
fetches them into ./wheels and checks the manifest lists exactly those files.

Usage (from the repository root):
	uv run --with pip tools/fetch_wheels.py
	blender --command extension build --split-platforms
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS_DIR = ROOT / "wheels"
MANIFEST = ROOT / "blender_manifest.toml"

#Blender 4.3 -> 5.0 ship Python 3.11, Blender 5.1+ ship Python 3.13
PYTHON_VERSIONS = ["3.11", "3.13"]

#Blender platform -> pip platform tags (most recent first, pip picks the best match)
PLATFORMS = {
	"windows-x64": ["win_amd64"],
	"macos-arm64": ["macosx_14_0_arm64", "macosx_11_0_arm64"],
	"macos-x64": ["macosx_13_0_x86_64", "macosx_10_12_x86_64"],
	"linux-x64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64"],
}

#numpy and certifi ship with Blender, packaging does not on Blender < 5
PURE_PACKAGES = ["laspy==2.7.0", "packaging==25.0"]
#pyogrio wheels are abi3 (cp311-abi3): the same file serves Python 3.11 and 3.13
BINARY_PACKAGES = ["lazrs==0.8.2", "pyproj==3.7.2", "pillow==13.0.0", "pyogrio==0.13.0"]


def pip_download(args):
	cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--disable-pip-version-check", "--no-deps", "--only-binary=:all:", "--dest", str(WHEELS_DIR)] + args
	subprocess.run(cmd, check=True)


def main():
	WHEELS_DIR.mkdir(exist_ok=True)

	pip_download(PURE_PACKAGES)
	for pyVersion in PYTHON_VERSIONS:
		for platformTags in PLATFORMS.values():
			platformArgs = [arg for tag in platformTags for arg in ("--platform", tag)]
			pip_download(["--python-version", pyVersion, "--implementation", "cp"] + platformArgs + BINARY_PACKAGES)

	downloaded = sorted(p.name for p in WHEELS_DIR.glob("*.whl"))
	listed = sorted(Path(p).name for p in re.findall(r'"(\./wheels/[^"]+\.whl)"', MANIFEST.read_text()))

	if downloaded != listed:
		print("blender_manifest.toml 'wheels' does not match ./wheels, expected:\n")
		print("wheels = [")
		for name in downloaded:
			print(f'  "./wheels/{name}",')
		print("]")
		sys.exit(1)
	print(f"{len(downloaded)} wheels in ./wheels match blender_manifest.toml")


if __name__ == "__main__":
	main()
