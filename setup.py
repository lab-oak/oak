from setuptools import setup, find_packages

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"oak": ["_native/*"]},
)
