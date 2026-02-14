import os
import sys
import subprocess
import signal


def _run_binary(binary_name, prefix="_bin"):
    bin_path = os.path.join(os.path.dirname(__file__), prefix, binary_name)

    if os.name != "nt":
        os.chmod(bin_path, 0o755)

    # Use subprocess.Popen so we can forward signals properly
    proc = subprocess.Popen([bin_path, *sys.argv[1:]])

    try:
        # Wait for the process to finish
        return_code = proc.wait()
    except KeyboardInterrupt:
        # Forward Ctrl+C to the child process
        if os.name == "nt":
            # On Windows, send CTRL_BREAK_EVENT (requires creationflags)
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # On Unix, send SIGINT
            proc.send_signal(signal.SIGINT)
        return_code = proc.wait()

    sys.exit(return_code)


def test():
    _run_binary("search-test")


def vs():
    _run_binary("vs")


def generate():
    _run_binary("generate")


def benchmark():
    _run_binary("benchmark")


def chall():
    _run_binary("chall")
