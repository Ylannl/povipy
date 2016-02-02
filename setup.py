from setuptools import setup, find_packages
setup(
    name = "pointvis",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["numpy>=1.9.1", "pyopengl", "glfw", "click", "pointio", "laspy"],
    dependency_links = ["git+https://github.com/Ylannl/pointio.git#egg=pointio"],
    # extras_require = {
    #     'LAS':  ["laspy"],
    #     'numba': ["numba"],
    #},
    scripts = ['pv.py']
)