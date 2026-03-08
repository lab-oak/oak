from setuptools import setup, find_packages
from setuptools import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def get_tag(self):
        return "py3", "none", "linux_x86_64"


directory = "Release"

setup(
    name="oaks-lab",
    version="1.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "oak": [
            f"_native/{directory}/*.so",
            f"_bin/{directory}/search-test",
            f"_bin/{directory}/generate",
            f"_bin/{directory}/vs",
            f"_bin/{directory}/chall",
            f"_bin/{directory}/benchmark",
        ],
    },
    distclass=BinaryDistribution,
)
