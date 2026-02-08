import os
import sys
import subprocess

def _run_binary(binary_name, prefix="_bin"):
    bin_path = os.path.join(os.path.dirname(__file__), prefix, binary_name)

    if os.name != "nt":
        os.chmod(bin_path, 0o755)

    sys.exit(subprocess.call([bin_path, *sys.argv[1:]]))


def vs():
    _run_binary("vs")

def generate():
    _run_binary("generate")

def benchmark():
    _run_binary("benchmark")

def chall():
    _run_binary("chall")

