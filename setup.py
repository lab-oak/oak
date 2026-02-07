from setuptools import setup, find_packages
from setuptools import Distribution

# Makes wheel platform-specific (because of .so)
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name="oak",
    version="1.0.2",
    packages=find_packages("src"),  # will find 'oak' now
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "oak": [
            "_native/*.so",  # include the .so files
            "_bin/*",        # include any binaries you want packaged
        ],
    },
    distclass=BinaryDistribution,
)
