from setuptools import setup, find_packages
setup(
    name = "povi",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["numpy>=1.9.1", "pyopengl", "glfw", "click", "pointio", "laspy"],
    dependency_links = ["https://github.com/Ylannl/pointio.git/tarbal/master#egg=pointio-0.1"],
    # extras_require = {
    #     'LAS':  ["laspy"],
    #     'numba': ["numba"],
    #},
    scripts = ['util/povilas.py', 'util/povimat.py']
)