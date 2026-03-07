from setuptools import setup, find_packages
from setuptools import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def get_tag(self):
        return "py3", "none", "linux_x86_64"


setup(
    name="oaks-lab",
    version="1.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "oak": [
            "_native/*.so",
            "_bin/search-test",
            "_bin/generate",
            "_bin/vs",
            "_bin/chall",
            "_bin/benchmark",
        ],
    },
    distclass=BinaryDistribution,
)
