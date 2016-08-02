from setuptools import setup, find_packages
setup(
    name = "povi",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["numpy>=1.9.1", "pyopengl", "click"],
    include_package_data = True
)