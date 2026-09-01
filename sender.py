import ipaddress
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RECEIVER_IP = "192.168.1.50"
PORT = 5000

WIDTH = 1280
HEIGHT = 720
FPS = 30

# H.264 bitrate in kbit/s
BITRATE = 4000

# Camera index
CAMERA_INDEX = 0

# Automatically try to install GStreamer if missing
AUTO_INSTALL = True


# ============================================================
# CONSOLE HELPERS
# ============================================================


def info(message):
    print(f"[INFO] {message}")


def success(message):
    print(f"[OK]   {message}")


def warning(message):
    print(f"[WARN] {message}")


def error(message):
    print(f"[ERROR] {message}")


def run_command(command, check=True):
    """
    Run a system command while showing it to the user.
    """

    if isinstance(command, list):
        printable = " ".join(str(x) for x in command)
    else:
        printable = command

    info(f"Running: {printable}")

    return subprocess.run(command, check=check)


# ============================================================
# GSTREAMER DISCOVERY
# ============================================================


def find_gstreamer_binary(binary_name):
    """
    Search PATH first, then common GStreamer installation locations.
    """

    system = platform.system()

    executable = f"{binary_name}.exe" if system == "Windows" else binary_name

    # --------------------------------------------------------
    # 1. Search PATH
    # --------------------------------------------------------

    found = shutil.which(executable)

    if found:
        return found

    # --------------------------------------------------------
    # 2. Windows common locations
    # --------------------------------------------------------

    if system == "Windows":
        possible_roots = []

        # Environment variables used by GStreamer
        for variable in [
            "GSTREAMER_1_0_ROOT_MSVC_X86_64",
            "GSTREAMER_1_0_ROOT_MSVC_X86",
            "GSTREAMER_1_0_ROOT_X86_64",
            "GSTREAMER_1_0_ROOT_X86",
        ]:
            value = os.environ.get(variable)

            if value:
                possible_roots.append(Path(value))

        possible_roots += [
            Path(r"C:\gstreamer\1.0\msvc_x86_64"),
            Path(r"C:\gstreamer\1.0\msvc_x86"),
            Path(r"C:\gstreamer\1.0\x86_64"),
            Path(r"C:\gstreamer\1.0\x86"),
        ]

        program_files = os.environ.get("ProgramFiles")

        if program_files:
            possible_roots += [
                Path(program_files) / "gstreamer" / "1.0" / "msvc_x86_64",
                Path(program_files) / "gstreamer" / "1.0" / "msvc_x86",
            ]

        for root in possible_roots:
            candidate = root / "bin" / executable

            if candidate.exists():
                # Add GStreamer bin directory to current process PATH
                os.environ["PATH"] = (
                    str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                )

                return str(candidate)

    # --------------------------------------------------------
    # 3. macOS common Homebrew locations
    # --------------------------------------------------------

    elif system == "Darwin":
        candidates = [
            Path("/opt/homebrew/bin") / executable,
            Path("/usr/local/bin") / executable,
        ]

        for candidate in candidates:
            if candidate.exists():
                os.environ["PATH"] = (
                    str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                )

                return str(candidate)

    # --------------------------------------------------------
    # 4. Linux common locations
    # --------------------------------------------------------

    elif system == "Linux":
        candidates = [
            Path("/usr/bin") / executable,
            Path("/usr/local/bin") / executable,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return None


# ============================================================
# GSTREAMER INSTALLATION
# ============================================================


def install_gstreamer_windows():
    info("Operating system: Windows")

    winget = shutil.which("winget")

    if not winget:
        raise RuntimeError(
            "winget is not installed.\nInstall/update Windows App Installer first."
        )

    info("Installing GStreamer using winget...")

    command = [
        winget,
        "install",
        "--exact",
        "--id",
        "gstreamerproject.gstreamer",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--silent",
    ]

    run_command(command)

    success("GStreamer installation command finished.")


def install_gstreamer_macos():
    info("Operating system: macOS")

    brew = shutil.which("brew")

    if not brew:
        raise RuntimeError(
            "Homebrew is not installed.\n"
            "Automatic GStreamer installation on macOS requires Homebrew.\n"
            "Install Homebrew first and run this script again."
        )

    info("Updating Homebrew package information...")

    run_command([brew, "update"], check=False)

    info("Installing GStreamer...")

    run_command([brew, "install", "gstreamer"])

    success("GStreamer installed.")


def install_gstreamer_linux():
    info("Operating system: Linux")

    apt = shutil.which("apt-get")

    if not apt:
        raise RuntimeError(
            "Automatic installation currently supports Debian/Ubuntu through apt-get."
        )

    info("Updating apt package list...")

    run_command(["sudo", apt, "update"])

    info("Installing GStreamer and required plugins...")

    packages = [
        "gstreamer1.0-tools",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-plugins-ugly",
        "gstreamer1.0-libav",
    ]

    run_command(["sudo", apt, "install", "-y", *packages])

    success("GStreamer installed.")


def install_gstreamer():
    system = platform.system()

    print()
    print("=" * 60)
    print("GStreamer installation")
    print("=" * 60)

    if system == "Windows":
        install_gstreamer_windows()

    elif system == "Darwin":
        install_gstreamer_macos()

    elif system == "Linux":
        install_gstreamer_linux()

    else:
        raise RuntimeError(f"Automatic installation is not supported on {system}.")


# ============================================================
# GSTREAMER CHECK
# ============================================================


def ensure_gstreamer():
    info("Checking GStreamer...")

    gst_launch = find_gstreamer_binary("gst-launch-1.0")

    gst_inspect = find_gstreamer_binary("gst-inspect-1.0")

    if gst_launch and gst_inspect:
        success("GStreamer found.")

        return gst_launch, gst_inspect

    warning("GStreamer was not found.")

    if not AUTO_INSTALL:
        raise RuntimeError("GStreamer is required but AUTO_INSTALL=False.")

    install_gstreamer()

    # Search again after installation
    gst_launch = find_gstreamer_binary("gst-launch-1.0")

    gst_inspect = find_gstreamer_binary("gst-inspect-1.0")

    if not gst_launch or not gst_inspect:
        raise RuntimeError(
            "\nGStreamer installation finished, but binaries "
            "could not be found.\n"
            "Restart the terminal/computer and run sender.py again."
        )

    success("GStreamer installation verified.")

    return gst_launch, gst_inspect


# ============================================================
# PLUGIN CHECK
# ============================================================


def plugin_exists(gst_inspect, plugin):
    result = subprocess.run(
        [gst_inspect, plugin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def get_camera_plugin():
    system = platform.system()

    if system == "Windows":
        return "ksvideosrc"

    if system == "Darwin":
        return "avfvideosrc"

    if system == "Linux":
        return "v4l2src"

    raise RuntimeError(f"Unsupported operating system: {system}")


def check_plugins(gst_inspect):
    camera_plugin = get_camera_plugin()

    required_plugins = [
        camera_plugin,
        "videoconvert",
        "videoscale",
        "videorate",
        "queue",
        "x264enc",
        "rtph264pay",
        "udpsink",
    ]

    print()
    print("=" * 60)
    print("Checking GStreamer plugins")
    print("=" * 60)

    missing = []

    for plugin in required_plugins:
        if plugin_exists(gst_inspect, plugin):
            success(plugin)

        else:
            error(plugin)
            missing.append(plugin)

    if not missing:
        success("All required plugins are installed.")
        return

    print()

    warning("Missing plugins: " + ", ".join(missing))

    # Attempt installation/repair
    if AUTO_INSTALL:
        info("Attempting to install/repair GStreamer plugins...")

        install_gstreamer()

        # Check again
        still_missing = []

        for plugin in missing:
            if not plugin_exists(gst_inspect, plugin):
                still_missing.append(plugin)

        if not still_missing:
            success("All missing plugins are now available.")

            return

        raise RuntimeError(
            "These required GStreamer plugins are still missing: "
            + ", ".join(still_missing)
        )

    raise RuntimeError("Required plugins are missing.")


# ============================================================
# VERSION CHECK
# ============================================================


def show_gstreamer_version(gst_launch):
    print()
    info("GStreamer version:")

    subprocess.run([gst_launch, "--version"])


# ============================================================
# NETWORK CONFIG CHECK
# ============================================================


def validate_network_config():
    print()
    print("=" * 60)
    print("Network configuration")
    print("=" * 60)

    try:
        ipaddress.ip_address(RECEIVER_IP)

    except ValueError:
        raise RuntimeError(f"Invalid receiver IP address: {RECEIVER_IP}")

    if not 1 <= PORT <= 65535:
        raise RuntimeError(f"Invalid UDP port: {PORT}")

    success(f"Receiver: {RECEIVER_IP}:{PORT}")


# ============================================================
# CAMERA SOURCE
# ============================================================


def get_camera_source():
    system = platform.system()

    if system == "Windows":
        return [
            "ksvideosrc",
            f"device-index={CAMERA_INDEX}",
        ]

    elif system == "Darwin":
        return [
            "avfvideosrc",
            f"device-index={CAMERA_INDEX}",
        ]

    elif system == "Linux":
        return [
            "v4l2src",
            f"device=/dev/video{CAMERA_INDEX}",
        ]

    raise RuntimeError(f"Unsupported operating system: {system}")


# ============================================================
# BUILD GSTREAMER PIPELINE
# ============================================================


def build_pipeline(gst_launch):
    camera = get_camera_source()

    pipeline = [
        gst_launch,
        "-e",
        # Camera
        *camera,
        "!",
        # Convert raw video
        "videoconvert",
        "!",
        # Allow resolution conversion
        "videoscale",
        "!",
        # Allow FPS conversion
        "videorate",
        "!",
        # Output format
        (f"video/x-raw,width={WIDTH},height={HEIGHT},framerate={FPS}/1"),
        "!",
        # Low latency queue
        "queue",
        "max-size-buffers=2",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        # H264 encoder
        "x264enc",
        "tune=zerolatency",
        "speed-preset=ultrafast",
        f"bitrate={BITRATE}",
        f"key-int-max={FPS}",
        "byte-stream=true",
        "!",
        # H264 output format
        "video/x-h264,profile=baseline",
        "!",
        # RTP packetization
        "rtph264pay",
        "config-interval=1",
        "pt=96",
        "!",
        # UDP output
        "udpsink",
        f"host={RECEIVER_IP}",
        f"port={PORT}",
        "sync=false",
        "async=false",
    ]

    return pipeline


# ============================================================
# START STREAM
# ============================================================


def start_stream(gst_launch):
    pipeline = build_pipeline(gst_launch)

    print()
    print("=" * 60)
    print("VIDEO STREAM")
    print("=" * 60)

    print(f"Receiver:   {RECEIVER_IP}:{PORT}")

    print(f"Resolution: {WIDTH}x{HEIGHT}")

    print(f"FPS:        {FPS}")

    print(f"Bitrate:    {BITRATE} kbit/s")

    print("Codec:      H.264")

    print("Transport:  RTP / UDP")

    print(f"Camera:     {CAMERA_INDEX}")

    print()

    info("Starting video stream...")
    info("Press CTRL+C to stop.")

    print()

    process = None

    try:
        process = subprocess.Popen(pipeline)

        exit_code = process.wait()

        if exit_code != 0:
            raise RuntimeError(f"GStreamer exited with code {exit_code}.")

    except KeyboardInterrupt:
        print()

        info("Stopping stream...")

        if process:
            process.terminate()

            try:
                process.wait(timeout=3)

            except subprocess.TimeoutExpired:
                process.kill()

        success("Stream stopped.")


# ============================================================
# PREFLIGHT
# ============================================================


def preflight():
    print()
    print("=" * 60)
    print("VIDEO SENDER PREFLIGHT")
    print("=" * 60)

    print(f"Operating system: {platform.system()} {platform.release()}")

    print(f"Python: {sys.version.split()[0]}")

    # Python version
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required.")

    success("Python version OK.")

    # Network configuration
    validate_network_config()

    # GStreamer
    gst_launch, gst_inspect = ensure_gstreamer()

    # Version
    show_gstreamer_version(gst_launch)

    # Required plugins
    check_plugins(gst_inspect)

    print()
    print("=" * 60)

    success("PRE-FLIGHT CHECK PASSED")

    print("=" * 60)

    return gst_launch


# ============================================================
# MAIN
# ============================================================


def main():
    try:
        gst_launch = preflight()

        start_stream(gst_launch)

    except KeyboardInterrupt:
        print()
        info("Stopped by user.")

    except Exception as exc:
        print()

        print("=" * 60)

        error(str(exc))

        print("=" * 60)

        sys.exit(1)


if __name__ == "__main__":
    main()
