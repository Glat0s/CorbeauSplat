import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from app.core.system import resolve_project_root

logger = logging.getLogger(__name__)

# Constants
EXTRACTOR_360_REPO = "https://github.com/nicolasdiolez/360Extractor"
BRUSH_REPO = "https://github.com/ArthurBrussee/brush.git"
SHARP_REPO = "https://github.com/apple/ml-sharp.git"
GLOMAP_REPO = "https://github.com/colmap/glomap.git"
SUPERPLAT_REPO = "https://github.com/playcanvas/supersplat.git"
REALESRGAN_PIP = "realesrgan"


class EngineDependency:
    """Represents an external engine (Colmap, Glomap, Brush, etc.)"""

    def __init__(self, name, repo_url=None, bin_name=None):
        self.name = name
        self.repo_url = repo_url
        self.bin_name = bin_name
        self.root = self.resolve_project_root()
        self.engines_dir = self.root / "engines"
        self.version_file = self.engines_dir / f"{name}.version"
        self.target_dir = self.engines_dir / name
        self.bin_path = self.engines_dir / (bin_name if bin_name else name)

    def resolve_project_root(self) -> Path:
        return resolve_project_root()

    def is_enabled_in_config(self, config: dict) -> bool:
        """[AUDIT] SOLID-OCP : Permet au moteur de decider s'il est actif"""
        return config.get(f"{self.name}_enabled", True)

    def is_installed(self) -> bool:
        return self.bin_path.exists()

    def get_local_version(self) -> str:
        if self.version_file.exists():
            return self.version_file.read_text().strip()
        return ""

    def save_local_version(self, version: str):
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        self.version_file.write_text(version)

    def get_remote_version(self) -> str:
        if not self.repo_url:
            return ""
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", self.repo_url, "HEAD"], text=True
            ).strip()
            return output.split()[0] if output else ""
        except Exception as e:
            logger.warning("Failed to get remote version for %s: %s", self.repo_url, e)
            return ""

    def update_git(self):
        """Clones or pulls the repository"""
        if not self.repo_url:
            return
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        if not self.target_dir.exists():
            logger.info("Cloning %s...", self.name)
            subprocess.check_call(["git", "clone", self.repo_url, str(self.target_dir)])
        else:
            logger.info("Updating %s...", self.name)
            subprocess.check_call(["git", "-C", str(self.target_dir), "pull"])

    def install(self):
        """Must be overridden"""
        raise NotImplementedError()

    def uninstall(self):
        """Standard uninstallation: remove target_dir and version file"""
        if self.target_dir.exists():
            logger.info("Removing %s...", self.target_dir)
            shutil.rmtree(str(self.target_dir))
        if self.version_file.exists():
            self.version_file.unlink()
        logger.info("%s uninstalled.", self.name)
        return True


class PipEngine(EngineDependency):
    """Engine installed via pip in a dedicated venv"""

    def __init__(self, name, repo_url, venv_name):
        super().__init__(name, repo_url)
        self.venv_dir = self.root / venv_name
        self.python_bin = (
            self.venv_dir
            / ("Scripts" if sys.platform == "win32" else "bin")
            / ("python.exe" if sys.platform == "win32" else "python")
        )
        self.bin_path = self.python_bin  # For pip engines, the python bin is the marker

    def is_installed(self) -> bool:
        return self.python_bin.exists()

    def create_venv(self, python_cmd=sys.executable):
        if self.venv_dir.exists() and not self.python_bin.exists():
            logger.warning(
                "Broken venv at %s (symlink or binary missing) — removing...", self.venv_dir
            )
            shutil.rmtree(str(self.venv_dir))

        if not self.venv_dir.exists():
            logger.info("Creating venv: %s", self.venv_dir)
            subprocess.check_call([python_cmd, "-m", "venv", str(self.venv_dir)])

        # Ensure pip is present (sometimes venv is created --without-pip on some systems)
        with contextlib.suppress(Exception):
            subprocess.check_call(
                [str(self.python_bin), "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Upgrade pip
        try:
            subprocess.check_call(
                [str(self.python_bin), "-m", "pip", "install", "--upgrade", "pip", "--no-input"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("Failed to upgrade pip in %s: %s", self.venv_dir, e)

    def pip_install(self, args, cwd=None):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.check_call(
            [str(self.python_bin), "-m", "pip", "install"]
            + args
            + ["--no-input", "--progress-bar", "off"],
            env=env,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def uninstall(self):
        """Remove venv and target_dir"""
        if self.venv_dir.exists():
            logger.info("Removing venv %s...", self.venv_dir)
            shutil.rmtree(str(self.venv_dir))
        return super().uninstall()


class DependencyManager:
    def __init__(self, engines_dir: Path):
        self.engines_dir = engines_dir
        self.engines = {}

    def register(self, engine: EngineDependency):
        self.engines[engine.name] = engine

    def get_config(self) -> dict:
        p = self.engines_dir.parent / "config.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    def main_install(self, check_only=False, startup=False):
        logger.info("--- System Dependency Check ---")
        install_system_dependencies(check_only=check_only or startup)

        config = self.get_config()
        missing_engines_startup = False

        for name, engine in self.engines.items():
            # [AUDIT] OCP : Le moteur decide s'il est active
            enabled = engine.is_enabled_in_config(config)

            # During --check or --startup, we audit everything. During install, we respect enablement.
            if not enabled and not (check_only or startup):
                continue

            remote = engine.get_remote_version()
            local = engine.get_local_version()
            # Normalize local version for comparison (strip build-mode suffixes like -source)
            local_clean = local.replace("-source", "") if local else local

            if not engine.is_installed():
                if check_only:
                    pass  # Just report status later
                elif startup:
                    logger.info("Auto-installing %s on startup...", name.capitalize())
                    try:
                        engine.install()
                        logger.info("%s installed automatically.", name.capitalize())
                    except Exception as e:
                        logger.error("Auto-install failed for %s: %s", name, e)
                else:
                    logger.info("Auto-installing missing engine [%s]...", name)
                    engine.install()

                # Report status for check/startup
                if not engine.is_installed():
                    status = f"  ❌ {name.capitalize()}: Missing"
                    if startup or check_only:
                        logger.info(status)
                    missing_engines_startup = True

            elif remote and local and remote != local_clean:
                # Update Available

                # Check Auto-Update Preference
                cfg_section = config.get("config", {})
                auto_update = config.get(f"{name}_auto_update", False) or cfg_section.get(
                    f"{name}_auto_update", False
                )

                if startup and auto_update:
                    logger.info("Auto-updating %s...", name.capitalize())
                    try:
                        engine.install()
                        logger.info("%s updated.", name.capitalize())
                    except Exception as e:
                        logger.error("Auto-update failed for %s: %s", name, e)
                elif check_only:
                    logger.warning(
                        "%s: update available (%s -> %s)", name.capitalize(), local_clean, remote
                    )
                else:
                    logger.info("Auto-updating %s (%s -> %s)...", name, local_clean, remote)
                    engine.install()
            else:
                if check_only:
                    logger.info("%s: ready.", name.capitalize())

        if missing_engines_startup:
            logger.info("Note: automatically installed missing engines.")


class Extractor360EngineDep(PipEngine):
    def __init__(self):
        super().__init__("extractor_360", EXTRACTOR_360_REPO, ".venv_360")
        self.script_path = self.target_dir / "src" / "main.py"

    def is_enabled_in_config(self, config: dict) -> bool:
        return config.get("extractor_360_params", {}).get("enabled", False) or config.get(
            "extractor_360_enabled", False
        )

    def install(self):
        self.update_git()
        self.create_venv()
        req_file = self.target_dir / "requirements.txt"
        if req_file.exists():
            self.pip_install(["-r", str(req_file)])
        self.save_local_version(self.get_remote_version())


class BrushEngineDep(EngineDependency):
    def __init__(self):
        super().__init__("brush", BRUSH_REPO)

    def is_enabled_in_config(self, config: dict) -> bool:
        return config.get("brush_params", {}).get("enabled", False) or config.get(
            "brush_enabled", False
        )

    def get_remote_version(self) -> str:
        """Queries GitHub API for the latest Brush release tag."""
        import json as _json
        import urllib.request

        try:
            req = urllib.request.Request(
                "https://api.github.com/repos/ArthurBrussee/brush/releases/latest",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "CorbeauSplat"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
                tag = data.get("tag_name", "")
                if tag:
                    logger.info("Latest Brush release: %s", tag)
                    return tag
        except Exception as e:
            logger.warning("Could not fetch latest Brush version: %s", e)
        return ""

    def _get_head_commit(self) -> str:
        """Returns the short HEAD commit hash of the remote repo."""
        try:
            out = subprocess.check_output(
                ["git", "ls-remote", self.repo_url, "HEAD"], text=True, timeout=10
            ).strip()
            return out.split()[0][:12] if out else ""
        except Exception as e:
            logger.warning("Could not fetch HEAD commit: %s", e)
            return ""

    def install(self):
        config = {}
        with contextlib.suppress(Exception):
            config = json.loads((self.root / "config.json").read_text())
        build_mode = config.get("brush_params", {}).get("build_mode", "release")

        if build_mode == "source":
            # Source mode tracks the HEAD commit, not a release tag
            remote_ref = self._get_head_commit()
            release_version = None
        else:
            remote_ref = self.get_remote_version() or "v0.3.0"
            release_version = remote_ref

        if self.bin_path.exists():
            local_ver = self.get_local_version()
            installed_as_source = "-source" in local_ver
            requested_source = build_mode == "source"

            if installed_as_source != requested_source:
                logger.info(
                    "Build mode changed (%s). Replacing existing binary...",
                    "release -> source" if requested_source else "source -> release",
                )
                self.bin_path.unlink()
                if self.version_file.exists():
                    self.version_file.unlink()
            else:
                # Same mode — compare versions
                local_ref = local_ver.replace("-source", "")
                if remote_ref and local_ref == remote_ref:
                    logger.info("Brush %s is already up to date.", local_ver)
                    return
                elif remote_ref:
                    logger.info("Brush update: %s -> %s. Updating...", local_ref, remote_ref)
                    self.bin_path.unlink()
                    if self.version_file.exists():
                        self.version_file.unlink()
                else:
                    logger.warning("Brush installed, could not check for updates.")
                    return

        if build_mode == "source":
            head = remote_ref or "HEAD"
            logger.info("Source mode selected — compiling from HEAD (%s)...", head[:7])
            if not self._install_from_source(head):
                logger.error(
                    "Source compilation failed. Relaunch after verifying your Rust/cargo installation."
                )
        else:
            logger.info("Release mode selected (%s). Downloading...", release_version)
            if not self._install_from_release(release_version):
                logger.error("Release download failed. Check your connection.")

    def _install_from_release(self, version: str) -> bool:
        import urllib.request
        import zipfile

        platform_suffix = "x86_64-pc-windows-msvc.zip"
        release_url = f"https://github.com/ArthurBrussee/brush/releases/download/{version}/brush-app-{platform_suffix}"
        logger.info("Downloading Brush %s from %s...", version, release_url)

        archive_path = self.engines_dir / f"brush-app-{platform_suffix}"
        try:
            urllib.request.urlretrieve(release_url, str(archive_path))
        except Exception as e:
            logger.warning("Download failed: %s", e)
            if archive_path.exists():
                archive_path.unlink()
            return False

        logger.info("Extracting Brush...")
        extract_dir = self.engines_dir / f"brush-extract-{version}"
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)  # nosec B202
        except Exception as e:
            logger.warning("Extraction failed: %s", e)
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(str(extract_dir), ignore_errors=True)
            return False
        finally:
            archive_path.unlink(missing_ok=True)

        # Find the executable anywhere in the extracted tree
        extracted_bin = None
        bin_names = {"brush-app", "brush_app", "brush-app.exe", "brush_app.exe"}
        for root_dir, _dirs, files in os.walk(str(extract_dir)):
            for f in files:
                if f in bin_names:
                    extracted_bin = Path(root_dir) / f
                    break
            if extracted_bin:
                break

        if not extracted_bin:
            logger.error("Could not find brush executable in archive.")
            shutil.rmtree(str(extract_dir), ignore_errors=True)
            return False

        dest = self.engines_dir / "brush"
        shutil.move(str(extracted_bin), str(dest))
        shutil.rmtree(str(extract_dir), ignore_errors=True)
        self.save_local_version(version)
        logger.info("Brush %s installed successfully from release binary.", version)
        return True

    def _install_from_source(self, head_ref: str) -> bool:
        """Compiles Brush from the latest commit on the default branch (HEAD)."""
        logger.info("Compiling Brush from source (HEAD: %s)...", head_ref[:7] if head_ref else "?")
        cargo = shutil.which("cargo")
        if not cargo:
            if not install_rust_toolchain():
                return False
            cargo = shutil.which("cargo")
            if not cargo:
                logger.error("cargo still not found after Rust install.")
                return False

        # Build from HEAD (no --tag), try --locked first then without
        base_cmd = [
            cargo,
            "install",
            "--git",
            self.repo_url,
            "brush-app",
            "--root",
            str(self.engines_dir),
        ]
        env = os.environ.copy()
        # Ensure cargo home bin is in PATH after potential rustup install
        cargo_bin = Path.home() / ".cargo" / "bin"
        if cargo_bin.exists():
            env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")

        success = False
        for extra in [["--locked"], []]:
            cmd = base_cmd + extra
            flag_str = " --locked" if extra else " (no lockfile)"
            logger.info("cargo install%s...", flag_str)
            try:
                subprocess.check_call(cmd, env=env)
                success = True
                break
            except subprocess.CalledProcessError as e:
                logger.warning("Attempt failed%s: %s", flag_str, e)

        if not success:
            logger.error("Brush source compilation failed.")
            return False

        bin_dir = self.engines_dir / "bin"
        moved = False
        for name in ["brush-app", "brush_app", "brush", "brush-app.exe", "brush_app.exe"]:
            src = bin_dir / name
            if src.exists():
                shutil.move(str(src), str(self.engines_dir / "brush"))
                moved = True
                break
        shutil.rmtree(str(bin_dir), ignore_errors=True)

        if not moved:
            logger.error("Binary not found after compilation.")
            return False

        # Save HEAD commit as version identifier
        version_str = f"{head_ref[:12]}-source" if head_ref else "HEAD-source"
        self.save_local_version(version_str)
        logger.info(
            "Brush compiled from HEAD (%s) and installed.", head_ref[:7] if head_ref else "?"
        )
        return True


class SharpEngineDep(PipEngine):
    def __init__(self):
        super().__init__("sharp", SHARP_REPO, ".venv_sharp")

    def is_enabled_in_config(self, config: dict) -> bool:
        return config.get("sharp_params", {}).get("enabled", False) or config.get(
            "sharp_enabled", False
        )

    def install(self):
        self.update_git()
        py_cmd = self._find_python_310_311()
        if not py_cmd:
            logger.error("Python 3.10-3.12 not found for Sharp. Install Python 3.11 and retry.")
            return

        self.create_venv(py_cmd)
        req_file = self.target_dir / "requirements.txt"
        if req_file.exists():
            loose = self.target_dir / "requirements_loose.txt"
            relax_requirements(str(req_file), str(loose))
            self.pip_install(["-r", str(loose)], cwd=str(self.target_dir))

        if (self.target_dir / "setup.py").exists() or (self.target_dir / "pyproject.toml").exists():
            self.pip_install(["-e", "."], cwd=str(self.target_dir))

        self.save_local_version(self.get_remote_version())

    def _find_python_310_311(self):
        """Finds Python 3.10, 3.11, or 3.12 on the current system."""
        if sys.platform == "win32":
            # Windows Python Launcher (py.exe) supports version selection
            for ver in ["3.11", "3.10", "3.12"]:
                try:
                    result = subprocess.run(
                        ["py", f"-{ver}", "-c", "import sys; print(sys.executable)"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except Exception:
                    pass
        # Unix / fallback: look for versioned python commands
        for cmd in ["python3.11", "python3.10", "python3.12", "python3", "python"]:
            p = shutil.which(cmd)
            if p:
                try:
                    ver_out = subprocess.check_output(
                        [p, "--version"], text=True, stderr=subprocess.STDOUT
                    ).strip()
                    parts = ver_out.split()
                    if len(parts) >= 2:
                        major, minor = parts[1].split(".")[:2]
                        if int(major) == 3 and int(minor) in (10, 11, 12):
                            return p
                except Exception:
                    pass
        return None


def load_config():
    """Loads config.json from project root/cwd"""
    p = Path("config.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def relax_requirements(src, dst):
    """Refactor utils: Relax strict torch deps"""
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("torch==") or line.strip().startswith("torchvision=="):
                line = line.replace("==", ">=")
            f_out.write(line)


def install_system_dependencies(check_only=False):
    return _install_system_dependencies_windows(check_only)


def _install_system_dependencies_windows(check_only=False):
    logger.info("--- System Dependency Check (Windows) ---")
    engines_dir = resolve_project_root() / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    colmap_missing = _find_colmap_in_engines(engines_dir) is None and (
        shutil.which("colmap") is None and shutil.which("colmap.exe") is None
    )
    ffmpeg_missing = shutil.which("ffmpeg") is None and shutil.which("ffmpeg.exe") is None

    missing = []
    if colmap_missing:
        missing.append("colmap")
    if ffmpeg_missing:
        missing.append("ffmpeg")

    if not missing:
        logger.info("All system dependencies present.")
        return True

    logger.warning("Missing system dependencies: %s", ", ".join(missing))
    if check_only:
        logger.info("Audit mode: automatic installation skipped.")
        return False

    if "colmap" in missing and not _download_colmap_windows(engines_dir):
        logger.warning("Download COLMAP manually from: https://github.com/colmap/colmap/releases")

    if "ffmpeg" in missing:
        installed = False
        if shutil.which("winget"):
            try:
                subprocess.check_call(
                    ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent"]
                )
                installed = True
            except Exception:
                pass
        if not installed and shutil.which("choco"):
            try:
                subprocess.check_call(["choco", "install", "ffmpeg", "-y"])
                installed = True
            except Exception:
                pass
        if not installed:
            logger.warning(
                "Download FFmpeg manually from: https://www.gyan.dev/ffmpeg/builds/ (add to PATH)"
            )

    return True


_COLMAP_WINDOWS_CUDA_URL = (
    "https://github.com/colmap/colmap/releases/download/4.0.1/colmap-x64-windows-cuda.zip"
)


def _find_colmap_in_engines(engines_dir: Path):
    """Finds colmap.exe inside engines/colmap/ directory tree."""
    colmap_dir = engines_dir / "colmap"
    if not colmap_dir.exists():
        return None
    for root_dir, _dirs, files in os.walk(str(colmap_dir)):
        for f in files:
            if f.lower() == "colmap.exe":
                return str(Path(root_dir) / f)
    return None


def _download_colmap_windows(engines_dir: Path) -> bool:
    """Downloads the COLMAP CUDA Windows binary."""
    import urllib.request
    import zipfile

    asset_url = _COLMAP_WINDOWS_CUDA_URL
    asset_name = asset_url.rsplit("/", 1)[-1]
    logger.info("Downloading COLMAP: %s...", asset_name)
    archive_path = engines_dir / asset_name
    try:
        urllib.request.urlretrieve(asset_url, str(archive_path))
    except Exception as e:
        logger.error("COLMAP download failed: %s", e)
        archive_path.unlink(missing_ok=True)
        return False

    logger.info("Extracting COLMAP...")
    colmap_dir = engines_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(colmap_dir)  # nosec B202
    except Exception as e:
        logger.error("COLMAP extraction failed: %s", e)
        archive_path.unlink(missing_ok=True)
        return False
    finally:
        archive_path.unlink(missing_ok=True)

    found = _find_colmap_in_engines(engines_dir)
    if found:
        logger.info("COLMAP installed: %s", found)
        return True
    logger.error("Could not find colmap.exe after extraction.")
    return False


def install_node_js():
    logger.info("Installing Node.js via winget...")
    try:
        subprocess.check_call(["winget", "install", "--id", "OpenJS.NodeJS.LTS", "-e", "--silent"])
        # Refresh PATH so node/npm are found in the current process
        node_dir = Path("C:/Program Files/nodejs")
        if node_dir.exists():
            os.environ["PATH"] = str(node_dir) + os.pathsep + os.environ.get("PATH", "")
        return True
    except Exception:
        logger.warning("Please install Node.js manually from https://nodejs.org/")
        return False


def _find_npm():
    """Finds npm executable, searching common Windows install locations if not in PATH."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm:
        return npm
    if sys.platform == "win32":
        # Search common Node.js install directories on Windows
        candidate_dirs = [
            Path("C:/Program Files/nodejs"),
            Path("C:/Program Files (x86)/nodejs"),
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "nodejs",
            Path(os.environ.get("APPDATA", "")) / "npm",
        ]
        for node_dir in candidate_dirs:
            for name in ["npm.cmd", "npm"]:
                p = node_dir / name
                if p.exists():
                    os.environ["PATH"] = str(node_dir) + os.pathsep + os.environ.get("PATH", "")
                    return str(p)
    return None


def install_build_tools():
    logger.info("Installing CMake and Ninja via winget...")
    try:
        subprocess.check_call(["winget", "install", "--id", "Kitware.CMake", "-e", "--silent"])
        subprocess.check_call(["winget", "install", "--id", "Ninja-build.Ninja", "-e", "--silent"])
        return True
    except Exception:
        logger.warning("Please install CMake and Ninja manually.")
        return False


# resolve_project_root is now imported from app.core.system


def uninstall_sharp():
    return SharpEngineDep().uninstall()


def install_sharp(engines_dir=None, version_file=None):
    # Compatibility wrapper
    dep = SharpEngineDep()
    dep.install()
    return dep.is_installed()


def uninstall_upscale():
    return UpscaleEngineDep().uninstall()


def install_upscale():
    dep = UpscaleEngineDep()
    dep.install()
    return dep.is_installed()


def uninstall_extractor_360():
    return Extractor360EngineDep().uninstall()


def install_extractor_360():
    dep = Extractor360EngineDep()
    dep.install()
    return dep.is_installed()


def get_venv_360_python():
    """Returns path to python executable in .venv_360"""
    root = resolve_project_root()
    if sys.platform == "win32":
        return root / ".venv_360" / "Scripts" / "python.exe"
    return root / ".venv_360" / "bin" / "python"


def get_remote_version(repo_url):
    """Gets the latest commit hash from the remote git repository"""
    try:
        output = subprocess.check_output(["git", "ls-remote", repo_url, "HEAD"], text=True).strip()
        if output:
            return output.split()[0]
    except Exception as e:
        logger.warning("Failed to verify remote version for %s: %s", repo_url, e)
    return None


def get_local_version(version_file: Path):
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except Exception:
            pass
    return None


def save_local_version(version_file: Path, version):
    if version:
        try:
            version_file.parent.mkdir(parents=True, exist_ok=True)
            version_file.write_text(version)
        except Exception as e:
            logger.warning("Failed to save local version: %s", e)


# --- CHECKERS ---


def check_cargo():
    return shutil.which("cargo") is not None


def check_node():
    return shutil.which("node") is not None and shutil.which("npm") is not None


def check_cmake_ninja():
    return shutil.which("cmake") is not None and shutil.which("ninja") is not None


# --- INSTALLERS HELPERS ---


def install_rust_toolchain():
    logger.info("Installing Rust (cargo)...")
    try:
        import tempfile
        import urllib.request

        rustup_url = "https://win.rustup.rs/x86_64"
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(rustup_url, tmp_path)
        subprocess.check_call([tmp_path, "-y", "--default-toolchain", "stable"])
        Path(tmp_path).unlink(missing_ok=True)

        cargo_bin = Path.home() / ".cargo" / "bin"
        if cargo_bin.exists():
            os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ["PATH"]
            logger.info("Rust installed and added to PATH.")
            return True
    except Exception as e:
        logger.error("Error installing Rust: %s", e)
    return False


class SuperSplatEngineDep(EngineDependency):
    def __init__(self):
        super().__init__("supersplat", SUPERPLAT_REPO)

    def install(self):
        if not shutil.which("node") and not install_node_js():
            return

        npm = _find_npm()
        if not npm:
            logger.error("npm not found. Install Node.js from https://nodejs.org/ and retry.")
            return

        # Reset local changes before pull to avoid conflicts (package-lock.json)
        if self.target_dir.exists():
            with contextlib.suppress(Exception):
                subprocess.check_call(
                    ["git", "-C", str(self.target_dir), "reset", "--hard", "HEAD"]
                )

        self.update_git()
        subprocess.check_call([npm, "install"], cwd=str(self.target_dir))
        subprocess.check_call([npm, "run", "build"], cwd=str(self.target_dir))
        self.save_local_version(self.get_remote_version())


class GlomapEngineDep(EngineDependency):
    def __init__(self):
        super().__init__("glomap", GLOMAP_REPO)
        # Fix: source code is in a separate dir, not replacing the binary
        self.target_dir = self.engines_dir / "glomap-source"

    def install(self):
        # GLOMAP has no pre-built Windows binaries; source build requires MSVC.
        # Skip on Windows — COLMAP's built-in exhaustive mapper is used as fallback.
        logger.info(
            "GLOMAP: no pre-built Windows binary available. COLMAP exhaustive mapper will be used instead."
        )


class UpscaleEngineDep(PipEngine):
    """Upscale is special as it installs in main sys.executable (usually)"""

    def __init__(self):
        # We use a fake venv name to satisfy PipEngine but we'll override
        super().__init__("upscale", None, "fake")
        self.python_bin = Path(sys.executable)

    def is_enabled_in_config(self, config: dict) -> bool:
        return config.get("upscale_params", {}).get("enabled", False) or config.get(
            "upscale_enabled", False
        )

    def is_installed(self) -> bool:
        from app.core.upscale_engine import UpscaleEngine

        return UpscaleEngine().is_installed()

    def install(self):
        # torch 2.8.0 / torchvision 0.23.0 — pinned pair for reproducibility
        if sys.platform == "win32":
            self.pip_install(
                [
                    "torch==2.8.0",
                    "torchvision==0.23.0",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu129",
                ]
            )
        else:
            self.pip_install(["torch==2.8.0", "torchvision==0.23.0"])
        pkgs = ["realesrgan==0.3.0", "kornia==0.8.2", "onnxruntime-gpu==1.24.3"]
        logger.info("Installing/updating: %s...", ", ".join(pkgs))
        self.pip_install(pkgs)


def main():
    root = Path(__file__).resolve().parent.parent.parent
    engines_dir = root / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    manager = DependencyManager(engines_dir)
    manager.register(GlomapEngineDep())
    manager.register(BrushEngineDep())
    manager.register(SharpEngineDep())
    manager.register(SuperSplatEngineDep())
    manager.register(UpscaleEngineDep())
    manager.register(Extractor360EngineDep())

    check_only = "--check" in sys.argv
    startup = "--startup" in sys.argv
    manager.main_install(check_only=check_only, startup=startup)


if __name__ == "__main__":
    main()
