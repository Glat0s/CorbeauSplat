import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from app.core.system import resolve_project_root

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
            print(f"Warning: Failed to get remote version for {self.repo_url}: {e}")
            return ""

    def update_git(self):
        """Clones or pulls the repository"""
        if not self.repo_url:
            return
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        if not self.target_dir.exists():
            print(f"Cloning {self.name}...")
            subprocess.check_call(["git", "clone", self.repo_url, str(self.target_dir)])
        else:
            print(f"Updating {self.name}...")
            subprocess.check_call(["git", "-C", str(self.target_dir), "pull"])

    def install(self):
        """Must be overridden"""
        raise NotImplementedError()

    def uninstall(self):
        """Standard uninstallation: remove target_dir and version file"""
        if self.target_dir.exists():
            print(f"Removing {self.target_dir}...")
            shutil.rmtree(str(self.target_dir))
        if self.version_file.exists():
            self.version_file.unlink()
        print(f"{self.name} uninstalled.")
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
            print(
                f"Broken venv detected at {self.venv_dir} (symlink or binary missing). Removing..."
            )
            shutil.rmtree(str(self.venv_dir))

        if not self.venv_dir.exists():
            print(f"Creating venv: {self.venv_dir}")
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
            print(f"Warning: Failed to upgrade pip in {self.venv_dir}: {e}")

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
            print(f"Removing venv {self.venv_dir}...")
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
        print("--- System Dependency Check ---")
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
                    print(f">>> Auto-installing {name.capitalize()} on startup...")
                    try:
                        engine.install()
                        print(f"✅ {name.capitalize()} installed automatically.")
                    except Exception as e:
                        print(f"❌ Auto-install failed for {name}: {e}")
                else:
                    print(f">>> Auto-installing missing engine [{name}]...")
                    engine.install()

                # Report status for check/startup
                if not engine.is_installed():
                    status = f"  ❌ {name.capitalize()}: Missing"
                    if startup or check_only:
                        print(status)
                    missing_engines_startup = True

            elif remote and local and remote != local_clean:
                # Update Available

                # Check Auto-Update Preference
                cfg_section = config.get("config", {})
                auto_update = config.get(f"{name}_auto_update", False) or cfg_section.get(
                    f"{name}_auto_update", False
                )

                if startup and auto_update:
                    print(f">>> Auto-updating {name.capitalize()}...")
                    try:
                        engine.install()
                        print(f"✅ {name.capitalize()} updated.")
                    except Exception as e:
                        print(f"❌ Auto-update failed for {name}: {e}")
                elif check_only:
                    print(f"  ⚠️  {name.capitalize()}: Update available ({local_clean} -> {remote})")
                else:
                    print(f">>> Auto-updating {name} ({local_clean} -> {remote})...")
                    engine.install()
            else:
                if check_only:
                    print(f"  ✅ {name.capitalize()}: Ready")

        if missing_engines_startup:
            print("\nℹ️  Note: Automatically installed missing engines.")


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
                    print(f"Latest Brush release: {tag}")
                    return tag
        except Exception as e:
            print(f"⚠️ Could not fetch latest Brush version: {e}")
        return ""

    def _get_head_commit(self) -> str:
        """Returns the short HEAD commit hash of the remote repo."""
        try:
            out = subprocess.check_output(
                ["git", "ls-remote", self.repo_url, "HEAD"], text=True, timeout=10
            ).strip()
            return out.split()[0][:12] if out else ""
        except Exception as e:
            print(f"⚠️ Could not fetch HEAD commit: {e}")
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
                print(
                    f"Build mode changed ({'release → source' if requested_source else 'source → release'}). Replacing existing binary..."
                )
                self.bin_path.unlink()
                if self.version_file.exists():
                    self.version_file.unlink()
            else:
                # Same mode — compare versions
                local_ref = local_ver.replace("-source", "")
                if remote_ref and local_ref == remote_ref:
                    print(f"Brush {local_ver} is already up to date.")
                    return
                elif remote_ref:
                    print(f"Brush update: {local_ref} → {remote_ref}. Updating...")
                    self.bin_path.unlink()
                    if self.version_file.exists():
                        self.version_file.unlink()
                else:
                    print("Brush installed, could not check for updates.")
                    return

        if build_mode == "source":
            head = remote_ref or "HEAD"
            print(f"Source mode selected — compiling from HEAD ({head[:7]})...")
            if not self._install_from_source(head):
                print(
                    "❌ Source compilation failed. Relaunch after verifying your Rust/cargo installation."
                )
        else:
            print(f"Release mode selected ({release_version}). Downloading...")
            if not self._install_from_release(release_version):
                print("❌ Release download failed. Check your connection.")

    def _install_from_release(self, version: str) -> bool:
        import platform
        import tarfile
        import urllib.request
        import zipfile

        system = platform.system()
        machine = platform.machine()

        platform_suffix = None
        if system == "Darwin" and machine == "arm64":
            platform_suffix = "aarch64-apple-darwin.tar.xz"
        elif system == "Windows" and machine == "AMD64":
            platform_suffix = "x86_64-pc-windows-msvc.zip"
        elif system == "Linux" and machine == "x86_64":
            platform_suffix = "x86_64-unknown-linux-gnu.tar.xz"

        if not platform_suffix:
            print(f"⚠️ No pre-built release for {system}/{machine}.")
            return False

        release_url = f"https://github.com/ArthurBrussee/brush/releases/download/{version}/brush-app-{platform_suffix}"
        print(f"Downloading Brush {version} from {release_url}...")

        archive_path = self.engines_dir / f"brush-app-{platform_suffix}"
        try:
            urllib.request.urlretrieve(release_url, str(archive_path))
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            if archive_path.exists():
                archive_path.unlink()
            return False

        print("Extracting Brush...")
        extract_dir = self.engines_dir / f"brush-extract-{version}"
        extract_dir.mkdir(exist_ok=True)
        try:
            if archive_path.name.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dir)  # nosec B202
            else:
                with tarfile.open(archive_path, "r:xz") as tf:
                    tf.extractall(extract_dir)  # nosec B202
        except Exception as e:
            print(f"⚠️ Extraction failed: {e}")
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
            print("⚠️ Could not find brush executable in archive.")
            shutil.rmtree(str(extract_dir), ignore_errors=True)
            return False

        dest = self.engines_dir / "brush"
        shutil.move(str(extracted_bin), str(dest))
        shutil.rmtree(str(extract_dir), ignore_errors=True)

        if system != "Windows":
            os.chmod(str(dest), 0o755)  # nosec B103

        self.save_local_version(version)
        print(f"✅ Brush {version} installed successfully from release binary.")
        return True

    def _install_from_source(self, head_ref: str) -> bool:
        """Compiles Brush from the latest commit on the default branch (HEAD)."""
        print(f"Compiling Brush from source (HEAD: {head_ref[:7] if head_ref else '?'})...")
        cargo = shutil.which("cargo")
        if not cargo:
            if not install_rust_toolchain():
                return False
            cargo = shutil.which("cargo")
            if not cargo:
                print("❌ cargo still not found after Rust install.")
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
            print(f"cargo install{flag_str}...")
            try:
                subprocess.check_call(cmd, env=env)
                success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Attempt failed{flag_str}: {e}")

        if not success:
            print("❌ Brush source compilation failed.")
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
            print("❌ Binary not found after compilation.")
            return False

        # Save HEAD commit as version identifier
        version_str = f"{head_ref[:12]}-source" if head_ref else "HEAD-source"
        self.save_local_version(version_str)
        print(f"✅ Brush compiled from HEAD ({head_ref[:7] if head_ref else '?'}) and installed.")
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
        # Sharp needs 3.11/3.10 ideally
        py311 = shutil.which("python3.11") or shutil.which("python3.10")
        if not py311:
            print("Python 3.11/3.10 missing for Sharp.")
            return

        self.create_venv(py311)
        req_file = self.target_dir / "requirements.txt"
        if req_file.exists():
            loose = self.target_dir / "requirements_loose.txt"
            relax_requirements(str(req_file), str(loose))
            self.pip_install(["-r", str(loose)], cwd=str(self.target_dir))

        if (self.target_dir / "setup.py").exists() or (self.target_dir / "pyproject.toml").exists():
            self.pip_install(["-e", "."], cwd=str(self.target_dir))

        self.save_local_version(self.get_remote_version())


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
    if sys.platform == "win32":
        return _install_system_dependencies_windows(check_only)
    return _install_system_dependencies_macos(check_only)


def _install_system_dependencies_windows(check_only=False):
    print("--- System Dependency Check (Windows) ---")
    missing = []
    for cmd in ["colmap", "ffmpeg"]:
        if shutil.which(cmd) is None and shutil.which(cmd + ".exe") is None:
            missing.append(cmd)

    if not missing:
        print("✅ System dependencies present.")
        return True

    print(f"Missing: {', '.join(missing)}")
    if check_only:
        print("ℹ️ Audit mode: automatic installation skipped.")
        return False

    # Try winget first (available on Windows 10/11)
    has_winget = shutil.which("winget") is not None
    # Try chocolatey as fallback
    has_choco = shutil.which("choco") is not None

    if not has_winget and not has_choco:
        print("ℹ️  No package manager found (winget/choco).")
        print("   Please install COLMAP and FFmpeg manually:")
        print("   COLMAP: https://github.com/colmap/colmap/releases")
        print("   FFmpeg: https://www.gyan.dev/ffmpeg/builds/ (add to PATH)")
        return False

    try:
        if has_winget:
            if "colmap" in missing:
                subprocess.check_call(
                    ["winget", "install", "--id", "UB-Mannheim.COLMAP", "-e", "--silent"]
                )
            if "ffmpeg" in missing:
                subprocess.check_call(
                    ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent"]
                )
        elif has_choco:
            if "colmap" in missing:
                subprocess.check_call(["choco", "install", "colmap", "-y"])
            if "ffmpeg" in missing:
                subprocess.check_call(["choco", "install", "ffmpeg", "-y"])
        return True
    except Exception as e:
        print(f"Automatic installation failed: {e}")
        print("Please install manually: COLMAP and FFmpeg, then add them to PATH.")
        return False


def _install_system_dependencies_macos(check_only=False):
    print("--- System Dependency Check (Homebrew) ---")
    missing = []
    for cmd in ["colmap", "ffmpeg"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)

    if sys.platform == "darwin":
        try:
            if (
                subprocess.run(
                    ["brew", "list", "libomp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ).returncode
                != 0
            ):
                missing.append("libomp")
            if (
                subprocess.run(
                    ["brew", "list", "freeimage"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                != 0
            ):
                missing.append("freeimage")
        except Exception:
            pass

    if not missing:
        print("✅ System dependencies present.")
        return True

    print(f"Missing: {', '.join(missing)}")
    if check_only:
        print("ℹ️ Audit mode: automatic installation skipped.")
        return False

    if shutil.which("brew") is None:
        print("ERROR: Homebrew required.")
        return False

    print("Installing via Homebrew...")
    try:
        if "colmap" in missing:
            subprocess.check_call(["brew", "install", "colmap"])
        if "ffmpeg" in missing:
            subprocess.check_call(["brew", "install", "ffmpeg"])
        if "libomp" in missing:
            subprocess.check_call(["brew", "install", "libomp"])
        if "freeimage" in missing:
            subprocess.check_call(["brew", "install", "freeimage"])
        return True
    except Exception:
        print("System installation failed.")
        return False


def install_node_js():
    if sys.platform == "win32":
        print("Installing Node.js via winget...")
        try:
            subprocess.check_call(
                ["winget", "install", "--id", "OpenJS.NodeJS.LTS", "-e", "--silent"]
            )
            return True
        except Exception:
            print("Please install Node.js manually from https://nodejs.org/")
            return False
    print("Installing Node.js via Homebrew...")
    try:
        subprocess.check_call(["brew", "install", "node"])
        return True
    except Exception:
        return False


def install_build_tools():
    if sys.platform == "win32":
        print("Installing CMake & Ninja via winget...")
        try:
            subprocess.check_call(["winget", "install", "--id", "Kitware.CMake", "-e", "--silent"])
            subprocess.check_call(
                ["winget", "install", "--id", "Ninja-build.Ninja", "-e", "--silent"]
            )
            return True
        except Exception:
            print("Please install CMake and Ninja manually.")
            return False
    print("Installing CMake & Ninja via Homebrew...")
    try:
        subprocess.check_call(["brew", "install", "cmake", "ninja"])
        return True
    except Exception:
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
        print(f"Attention: Impossible de verifier la version distante pour {repo_url}: {e}")
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
            print(f"Attention: Impossible d'enregistrer la version locale: {e}")


# --- CHECKERS ---


def check_cargo():
    return shutil.which("cargo") is not None


def check_brew():
    return shutil.which("brew") is not None


def check_node():
    return shutil.which("node") is not None and shutil.which("npm") is not None


def check_cmake_ninja():
    return shutil.which("cmake") is not None and shutil.which("ninja") is not None


def check_xcode_tools():
    """Checks if Xcode Command Line Tools are installed (macOS only)"""
    if sys.platform != "darwin":
        return True
    try:
        # xcode-select -p prints the path if installed, or exits with error
        subprocess.check_call(
            ["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


# --- INSTALLERS HELPERS ---


def install_rust_toolchain():
    print("Installing Rust (cargo)...")
    try:
        if sys.platform == "win32":
            # On Windows, download and run rustup-init.exe
            import tempfile
            import urllib.request

            rustup_url = "https://win.rustup.rs/x86_64"
            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
                tmp_path = tmp.name
            urllib.request.urlretrieve(rustup_url, tmp_path)
            subprocess.check_call([tmp_path, "-y", "--default-toolchain", "stable"])
            Path(tmp_path).unlink(missing_ok=True)
        else:
            # Install rustup non-interactively on Unix
            subprocess.check_call(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
            )

        # Add cargo bin to current PATH for this session
        cargo_bin = Path.home() / ".cargo" / "bin"
        if cargo_bin.exists():
            os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ["PATH"]
            print("Rust installed and added to PATH.")
            return True
    except Exception as e:
        print(f"Error installing Rust: {e}")
    return False


class SuperSplatEngineDep(EngineDependency):
    def __init__(self):
        super().__init__("supersplat", SUPERPLAT_REPO)

    def install(self):
        if not shutil.which("node") and not install_node_js():
            return

        # Reset local changes before pull to avoid conflicts (package-lock.json)
        if self.target_dir.exists():
            with contextlib.suppress(Exception):
                subprocess.check_call(
                    ["git", "-C", str(self.target_dir), "reset", "--hard", "HEAD"]
                )

        self.update_git()
        subprocess.check_call(["npm", "install"], cwd=str(self.target_dir))
        subprocess.check_call(["npm", "run", "build"], cwd=str(self.target_dir))
        self.save_local_version(self.get_remote_version())


class GlomapEngineDep(EngineDependency):
    def __init__(self):
        super().__init__("glomap", GLOMAP_REPO)
        # Fix: source code is in a separate dir, not replacing the binary
        self.target_dir = self.engines_dir / "glomap-source"

    def install(self):
        if sys.platform == "darwin" and not check_xcode_tools():
            print("Xcode Command Line Tools required.")
            return

        if not check_cmake_ninja() and not install_build_tools():
            return

        self.update_git()
        # Source dir is now handled by update_git via self.target_dir
        source_dir = self.target_dir

        build_dir = source_dir / "build"
        # Fix CMakeCache error by cleaning build dir if it exists
        if build_dir.exists():
            shutil.rmtree(str(build_dir))
        build_dir.mkdir(exist_ok=True)

        cmake_args = ["cmake", "..", "-GNinja", "-DCMAKE_BUILD_TYPE=Release"]
        env = os.environ.copy()

        if sys.platform == "darwin":
            try:
                libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
                include_p = f"{libomp}/include"
                lib_p = f"{libomp}/lib"
                cmake_args.extend(
                    [
                        f"-DOpenMP_ROOT={libomp}",
                        "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp",
                        "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp",
                    ]
                )
                env["LDFLAGS"] = f"-L{lib_p} -lomp"
                env["CPPFLAGS"] = f"-I{include_p} -Xpreprocessor -fopenmp"
            except Exception:
                pass

        subprocess.check_call(cmake_args, cwd=str(build_dir), env=env)
        subprocess.check_call(["ninja"], cwd=str(build_dir), env=env)

        # Binary name is glomap
        built_bin = None
        for p in [build_dir / "glomap" / "glomap", build_dir / "glomap"]:
            if p.exists() and not p.is_dir():
                built_bin = p
                break

        if built_bin:
            shutil.copy2(str(built_bin), str(self.engines_dir / "glomap"))
            self.save_local_version(self.get_remote_version())


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
        print(f"Installing/Updating: {', '.join(pkgs)}...")
        self.pip_install(pkgs)


class VR180EngineDep(PipEngine):
    """VR 180 green-screen segmentation engine (SAM + OpenCV + torch)"""

    def __init__(self):
        super().__init__("vr180", None, ".venv_vr180")

    def is_enabled_in_config(self, config: dict) -> bool:
        return config.get("vr180_params", {}).get("enabled", False) or config.get(
            "vr180_enabled", False
        )

    def install(self):
        self.create_venv()
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
            # triton-windows enables torch.compile on Windows
            self.pip_install(["triton-windows==3.6.0.post26"])
        else:
            self.pip_install(["torch==2.8.0", "torchvision==0.23.0"])
        self.pip_install(
            [
                "opencv-python==4.13.0.92",
                "numpy==2.4.3",
                "kornia==0.8.2",
                "onnxruntime-gpu==1.24.3",
            ]
        )
        # Install SAM (Segment Anything Model) from Meta
        self.pip_install(["git+https://github.com/facebookresearch/segment-anything.git"])
        self.save_local_version("sam-installed")
        print("✅ VR180 engine (SAM + OpenCV + kornia + onnxruntime-gpu) installed.")

    def is_installed(self) -> bool:
        if not self.python_bin.exists():
            return False
        try:
            result = subprocess.run(
                [
                    str(self.python_bin),
                    "-c",
                    "import cv2, torch; from segment_anything import sam_model_registry",
                ],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


def uninstall_vr180():
    return VR180EngineDep().uninstall()


def install_vr180():
    dep = VR180EngineDep()
    dep.install()
    return dep.is_installed()


def get_venv_vr180_python():
    """Returns path to python executable in .venv_vr180"""
    root = resolve_project_root()
    if sys.platform == "win32":
        return root / ".venv_vr180" / "Scripts" / "python.exe"
    return root / ".venv_vr180" / "bin" / "python"


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
    manager.register(VR180EngineDep())

    check_only = "--check" in sys.argv
    startup = "--startup" in sys.argv
    manager.main_install(check_only=check_only, startup=startup)


if __name__ == "__main__":
    main()
