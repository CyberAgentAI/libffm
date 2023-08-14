import os
from setuptools import Extension, find_packages, setup


def make_extensions():
    import numpy
    try:
        from Cython.Build import cythonize
        ext = ".pyx"
    except ImportError:
        cythonize = None
        ext = ".cpp"
    ext_modules = [
        Extension(
            "ffm.libffm",
            extra_compile_args=["-Wall", "-O3", "-std=c++0x", "-march=native", "-DUSESSE"],
            sources=[os.path.join("ffm", "libffm" + ext), "ffm.cpp"],
            include_dirs=[".", numpy.get_include()],
            language="c++",
        )
    ]
    if cythonize is not None:
        ext_modules = cythonize(ext_modules)
    return ext_modules


setup(
    name="ffm",
    version="0.4.1",
    description="LibFFM Python Package",
    long_description="LibFFM Python Package",
    install_requires=["numpy"],
    ext_modules=make_extensions(),
    maintainer="",
    maintainer_email="",
    zip_safe=False,
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={
        "console_scripts": [
            "pyffm-train = ffm.cli:ffm_train",
            "pyffm-predict = ffm.cli:ffm_predict",
        ],
    },
    include_package_data=False,
    data_files=[("", ["ffm.cpp", "ffm.h", "COPYRIGHT"])],
    package_data={"ffm": ["*.cpp", "*.h", "*.pyx", "*.pxd"]},
)
