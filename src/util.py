import sys
import os


def find_data_files(root_dir, ext):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                full_path = os.path.join(dirpath, filename)
                files.append(full_path)
    files.sort(key=os.path.getctime, reverse=True)
    return files


def save_args(namespace, path):
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "args")

    with open(out_path, "w") as f:
        for key, value in vars(namespace).items():
            f.write(f"--{key}={value}\n")
