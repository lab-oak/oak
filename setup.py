from setuptools import setup, find_packages
from setuptools import Distribution


# Makes wheel platform-specific (because of .so)
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="oaks-lab",
    version="1.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "oak": [
            "_native/*.so",
        ],
    },
    data_files=[
        (
            "oak/_bin",
            [
                "src/_bin/generate",
                "src/_bin/vs",
                "src/_bin/chall",
                "src/_bin/benchmark",
            ],
        ),
    ],
    distclass=BinaryDistribution,
)
