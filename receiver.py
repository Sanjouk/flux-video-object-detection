import importlib.util
import ipaddress
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# UDP port must be the same as in sender.py
PORT = 5000

# Resolution that Python / YOLO will receive.
# It does NOT have to match the original camera resolution:
# GStreamer will resize incoming video if necessary.
WIDTH = 1280
HEIGHT = 720

# YOLO model
#
# yolo11n.pt - fastest / lightest
# yolo11s.pt - more accurate
# yolo11m.pt - considerably heavier
#
MODEL_PATH = "yolo11n.pt"

# YOLO inference resolution
YOLO_IMAGE_SIZE = 640

# Minimum confidence
CONFIDENCE = 0.40

# Show resulting video
SHOW_VIDEO = True

# Automatically install missing software
AUTO_INSTALL = True

# RTP jitter buffer.
# Smaller = lower latency.
# Bigger = more resistant to unstable Wi-Fi.
RTP_LATENCY_MS = 50


# ============================================================
# CONSOLE HELPERS
# ============================================================


def info(message):
    print(f"[INFO]  {message}")


def success(message):
    print(f"[OK]    {message}")


def warning(message):
    print(f"[WARN]  {message}")


def error(message):
    print(f"[ERROR] {message}")


# ============================================================
# PYTHON PACKAGE INSTALLATION
# ============================================================

PYTHON_PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "ultralytics": "ultralytics",
    "torch": "torch",
}


def module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None


def install_python_package(package):
    info(f"Installing Python package: {package}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", package]
    )


def ensure_python_packages():
    print()
    print("=" * 70)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 70)

    missing = []

    for module, package in PYTHON_PACKAGES.items():
        if module_exists(module):
            success(f"{package}")
        else:
            warning(f"{package} is missing")
            missing.append(package)

    if not missing:
        return

    if not AUTO_INSTALL:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing))

    print()
    info("Installing missing Python packages...")

    # Upgrade pip first
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False
    )

    for package in missing:
        install_python_package(package)

    # Recheck
    failed = []

    for module, package in PYTHON_PACKAGES.items():
        if not module_exists(module):
            failed.append(package)

    if failed:
        raise RuntimeError("Could not install: " + ", ".join(failed))

    success("All required Python packages are installed.")


# ============================================================
# GSTREAMER DISCOVERY
# ============================================================


def find_gstreamer_binary(binary_name):
    system = platform.system()

    executable = f"{binary_name}.exe" if system == "Windows" else binary_name

    # --------------------------------------------------------
    # PATH
    # --------------------------------------------------------

    found = shutil.which(executable)

    if found:
        return found

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    if system == "Windows":
        possible_roots = []

        env_variables = [
            "GSTREAMER_1_0_ROOT_MSVC_X86_64",
            "GSTREAMER_1_0_ROOT_MSVC_X86",
            "GSTREAMER_1_0_ROOT_X86_64",
            "GSTREAMER_1_0_ROOT_X86",
        ]

        for variable in env_variables:
            value = os.environ.get(variable)

            if value:
                possible_roots.append(Path(value))

        possible_roots.extend(
            [
                Path(r"C:\gstreamer\1.0\msvc_x86_64"),
                Path(r"C:\gstreamer\1.0\msvc_x86"),
                Path(r"C:\gstreamer\1.0\x86_64"),
                Path(r"C:\gstreamer\1.0\x86"),
            ]
        )

        program_files = os.environ.get("ProgramFiles")

        if program_files:
            possible_roots.extend(
                [
                    Path(program_files) / "gstreamer" / "1.0" / "msvc_x86_64",
                    Path(program_files) / "gstreamer" / "1.0" / "msvc_x86",
                ]
            )

        for root in possible_roots:
            candidate = root / "bin" / executable

            if candidate.exists():
                os.environ["PATH"] = (
                    str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                )

                return str(candidate)

    # --------------------------------------------------------
    # MAC
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
    # LINUX
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
# GSTREAMER INSTALL
# ============================================================


def install_gstreamer_windows():
    winget = shutil.which("winget")

    if not winget:
        raise RuntimeError(
            "winget was not found.\nInstall Microsoft App Installer first."
        )

    info("Installing GStreamer through winget...")

    subprocess.check_call(
        [
            winget,
            "install",
            "--exact",
            "--id",
            "gstreamerproject.gstreamer",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    )


def install_gstreamer_macos():
    brew = shutil.which("brew")

    if not brew:
        raise RuntimeError(
            "Homebrew is required for automatic GStreamer installation on macOS."
        )

    info("Installing GStreamer...")

    subprocess.check_call([brew, "install", "gstreamer"])


def install_gstreamer_linux():
    apt = shutil.which("apt-get")

    if not apt:
        raise RuntimeError(
            "Automatic Linux installation currently supports Debian/Ubuntu."
        )

    subprocess.check_call(["sudo", apt, "update"])

    packages = [
        "gstreamer1.0-tools",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-plugins-ugly",
        "gstreamer1.0-libav",
    ]

    subprocess.check_call(["sudo", apt, "install", "-y", *packages])


def install_gstreamer():
    system = platform.system()

    if system == "Windows":
        install_gstreamer_windows()

    elif system == "Darwin":
        install_gstreamer_macos()

    elif system == "Linux":
        install_gstreamer_linux()

    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


# ============================================================
# GSTREAMER CHECK
# ============================================================


def ensure_gstreamer():
    print()
    print("=" * 70)
    print("CHECKING GSTREAMER")
    print("=" * 70)

    gst_launch = find_gstreamer_binary("gst-launch-1.0")

    gst_inspect = find_gstreamer_binary("gst-inspect-1.0")

    if gst_launch and gst_inspect:
        success("GStreamer found.")

        return gst_launch, gst_inspect

    warning("GStreamer is not installed.")

    if not AUTO_INSTALL:
        raise RuntimeError("GStreamer is required.")

    install_gstreamer()

    gst_launch = find_gstreamer_binary("gst-launch-1.0")

    gst_inspect = find_gstreamer_binary("gst-inspect-1.0")

    if not gst_launch or not gst_inspect:
        raise RuntimeError(
            "GStreamer installation completed, "
            "but executables cannot be found.\n"
            "Restart the terminal or computer and "
            "run receiver.py again."
        )

    success("GStreamer successfully installed.")

    return gst_launch, gst_inspect


# ============================================================
# GSTREAMER PLUGIN CHECK
# ============================================================


def plugin_exists(gst_inspect, plugin):
    result = subprocess.run(
        [gst_inspect, plugin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def check_gstreamer_plugins(gst_inspect):
    required_plugins = [
        "udpsrc",
        "rtpjitterbuffer",
        "rtph264depay",
        "h264parse",
        "avdec_h264",
        "videoconvert",
        "videoscale",
        "queue",
        "fdsink",
    ]

    print()
    print("=" * 70)
    print("CHECKING GSTREAMER PLUGINS")
    print("=" * 70)

    missing = []

    for plugin in required_plugins:
        if plugin_exists(gst_inspect, plugin):
            success(plugin)

        else:
            error(plugin)
            missing.append(plugin)

    if missing:
        raise RuntimeError("Missing GStreamer plugins:\n" + "\n".join(missing))

    success("All required GStreamer plugins are installed.")


# ============================================================
# DEVICE SELECTION
# ============================================================


def choose_yolo_device():
    import torch

    print()
    print("=" * 70)
    print("AI DEVICE")
    print("=" * 70)

    # NVIDIA CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

        success(f"NVIDIA CUDA available: {device_name}")

        return 0

    # Apple Silicon / Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        success("Apple Metal / MPS available.")

        return "mps"

    warning("GPU acceleration not available. Using CPU.")

    return "cpu"


# ============================================================
# NETWORK VALIDATION
# ============================================================


def check_network_configuration():
    if PORT < 1 or PORT > 65535:
        raise RuntimeError(f"Invalid UDP port: {PORT}")

    success(f"Listening on UDP port {PORT}")


# ============================================================
# BUILD RECEIVER PIPELINE
# ============================================================


def build_receiver_pipeline(gst_launch):
    caps = (
        "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"
    )

    raw_caps = f"video/x-raw,format=BGR,width={WIDTH},height={HEIGHT}"

    pipeline = [
        gst_launch,
        "-q",
        # ----------------------------------------------------
        # UDP INPUT
        # ----------------------------------------------------
        "udpsrc",
        f"port={PORT}",
        f"caps={caps}",
        "!",
        # ----------------------------------------------------
        # SMALL RTP JITTER BUFFER
        # ----------------------------------------------------
        "rtpjitterbuffer",
        f"latency={RTP_LATENCY_MS}",
        "drop-on-latency=true",
        "!",
        # ----------------------------------------------------
        # RTP → H264
        # ----------------------------------------------------
        "rtph264depay",
        "!",
        "h264parse",
        "!",
        # ----------------------------------------------------
        # H264 → RAW VIDEO
        # ----------------------------------------------------
        "avdec_h264",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        raw_caps,
        "!",
        # ----------------------------------------------------
        # PREVENT VIDEO BUFFERING
        # ----------------------------------------------------
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        # ----------------------------------------------------
        # SEND RAW BGR DATA TO PYTHON STDOUT
        # ----------------------------------------------------
        "fdsink",
        "fd=1",
        "sync=false",
    ]

    return pipeline


# ============================================================
# EXACT BYTE READER
# ============================================================


def read_exact(stream, byte_count):
    data = bytearray()

    while len(data) < byte_count:
        chunk = stream.read(byte_count - len(data))

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


# ============================================================
# LATEST FRAME STORAGE
# ============================================================


class LatestFrame:
    def __init__(self):
        self.frame = None

        self.frame_id = 0

        self.lock = threading.Lock()

        self.running = True

    def set(self, frame):
        with self.lock:
            self.frame = frame

            self.frame_id += 1

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id

            return (self.frame.copy(), self.frame_id)

    def stop(self):
        self.running = False


# ============================================================
# VIDEO READER THREAD
# ============================================================


def video_reader(process, latest_frame):
    import numpy as np

    frame_bytes = WIDTH * HEIGHT * 3

    info(f"Raw frame size: {frame_bytes:,} bytes")

    while latest_frame.running:
        raw_frame = read_exact(process.stdout, frame_bytes)

        if raw_frame is None:
            warning("GStreamer video stream stopped.")

            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8)

        frame = frame.reshape((HEIGHT, WIDTH, 3))

        # Store only the newest frame
        latest_frame.set(frame)

    latest_frame.stop()


# ============================================================
# LOAD YOLO
# ============================================================


def load_yolo():
    from ultralytics import YOLO

    print()
    print("=" * 70)
    print("LOADING YOLO")
    print("=" * 70)

    info(f"Model: {MODEL_PATH}")

    info(
        "If the model is not available locally, "
        "Ultralytics may download it automatically."
    )

    model = YOLO(MODEL_PATH)

    success("YOLO model loaded.")

    return model


# ============================================================
# YOLO PROCESSING LOOP
# ============================================================


def process_video(model, device, latest_frame):
    import cv2

    print()
    print("=" * 70)
    print("YOLO DETECTION STARTED")
    print("=" * 70)

    print(f"Model:       {MODEL_PATH}")

    print(f"YOLO size:   {YOLO_IMAGE_SIZE}")

    print(f"Confidence:  {CONFIDENCE}")

    print(f"Device:      {device}")

    print()

    info("Waiting for video stream...")

    last_frame_id = -1

    processed_frames = 0

    fps_start = time.time()

    display_fps = 0.0

    stream_started = False

    while latest_frame.running:
        frame, frame_id = latest_frame.get()

        # No frame received yet
        if frame is None:
            time.sleep(0.01)
            continue

        if not stream_started:
            success("Video stream received.")

            stream_started = True

        # YOLO already processed this frame
        if frame_id == last_frame_id:
            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=CONFIDENCE,
            device=device,
            verbose=False,
        )

        result = results[0]

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        processed_frames += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:
            display_fps = processed_frames / elapsed

            processed_frames = 0

            fps_start = time.time()

        # ----------------------------------------------------
        # DETECTED OBJECTS
        # ----------------------------------------------------

        boxes = result.boxes

        object_count = len(boxes) if boxes is not None else 0

        # ----------------------------------------------------
        # DRAW YOLO RESULT
        # ----------------------------------------------------

        annotated_frame = result.plot()

        cv2.putText(
            annotated_frame,
            f"YOLO FPS: {display_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Objects: {object_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if SHOW_VIDEO:
            cv2.imshow("Network YOLO Receiver", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Q or ESC
            if key == ord("q") or key == 27:
                info("Exit requested.")

                break

    cv2.destroyAllWindows()


# ============================================================
# START GSTREAMER
# ============================================================


def start_gstreamer(gst_launch):
    pipeline = build_receiver_pipeline(gst_launch)

    print()
    print("=" * 70)
    print("STARTING VIDEO RECEIVER")
    print("=" * 70)

    print(f"UDP port:       {PORT}")

    print(f"Output size:    {WIDTH}x{HEIGHT}")

    print(f"RTP latency:    {RTP_LATENCY_MS} ms")

    print("Codec:          H.264")

    print("Transport:      RTP / UDP")

    print()

    process = subprocess.Popen(
        pipeline,
        stdout=subprocess.PIPE,
        # Keep diagnostic messages visible
        stderr=None,
        # Large pipe buffer when supported
        bufsize=10 * 1024 * 1024,
    )

    return process


# ============================================================
# PREFLIGHT
# ============================================================


def preflight():
    print()
    print("=" * 70)
    print("YOLO VIDEO RECEIVER PREFLIGHT")
    print("=" * 70)

    print(f"OS:      {platform.system()} {platform.release()}")

    print(f"Python:  {sys.version.split()[0]}")

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required.")

    success("Python version OK.")

    # --------------------------------------------------------
    # Python packages
    # --------------------------------------------------------

    ensure_python_packages()

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    check_network_configuration()

    # --------------------------------------------------------
    # GStreamer
    # --------------------------------------------------------

    gst_launch, gst_inspect = ensure_gstreamer()

    check_gstreamer_plugins(gst_inspect)

    # --------------------------------------------------------
    # AI device
    # --------------------------------------------------------

    device = choose_yolo_device()

    print()
    print("=" * 70)

    success("PRE-FLIGHT CHECK PASSED")

    print("=" * 70)

    return (gst_launch, device)


# ============================================================
# MAIN
# ============================================================


def main():
    process = None

    latest_frame = None

    try:
        # ----------------------------------------------------
        # CHECK EVERYTHING
        # ----------------------------------------------------

        gst_launch, device = preflight()

        # ----------------------------------------------------
        # LOAD YOLO
        # ----------------------------------------------------

        model = load_yolo()

        # ----------------------------------------------------
        # START GSTREAMER RECEIVER
        # ----------------------------------------------------

        process = start_gstreamer(gst_launch)

        # ----------------------------------------------------
        # FRAME STORAGE
        # ----------------------------------------------------

        latest_frame = LatestFrame()

        # ----------------------------------------------------
        # VIDEO READER THREAD
        # ----------------------------------------------------

        reader_thread = threading.Thread(
            target=video_reader, args=(process, latest_frame), daemon=True
        )

        reader_thread.start()

        # ----------------------------------------------------
        # YOLO PROCESSING
        # ----------------------------------------------------

        process_video(model, device, latest_frame)

    except KeyboardInterrupt:
        print()

        info("Stopped by user.")

    except Exception as exc:
        print()

        print("=" * 70)

        error(str(exc))

        print("=" * 70)

    finally:
        if latest_frame:
            latest_frame.stop()

        if process:
            info("Stopping GStreamer...")

            process.terminate()

            try:
                process.wait(timeout=3)

            except subprocess.TimeoutExpired:
                process.kill()

        try:
            import cv2

            cv2.destroyAllWindows()

        except Exception:
            pass

        success("Receiver stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
