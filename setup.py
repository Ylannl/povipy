from setuptools import setup, find_packages
setup(
    name = "povi",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["numpy>=1.9.1", "pyopengl", "click", "pointio==0.1"],
    dependency_links = ["http://github.com/Ylannl/pointio/tarball/master#egg=pointio-0.1"],
    include_package_data = True,
    # extras_require = {
    #     'LAS':  ["laspy"]
    #},
    scripts = ['util/povimat.py']
)