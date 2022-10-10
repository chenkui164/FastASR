import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages
from pathlib import Path
import platform


requirements = []


def is_windows():
    return platform.system() == "Windows"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        print('***************************')
        print('ext is {}'.format(ext))
        print('ext.sourcedir is {}'.format(ext.sourcedir))
        print('self.build_temp is {}'.format(self.build_temp))
        print('self.build_lib is {}'.format(self.build_lib))
        print('extdir is {}'.format(extdir))
        print('os.path.sep is {}'.format(os.path.sep))
        print('self.compiler.compiler_type is {}'.format(
            self.compiler.compiler_type))
        print('self.parallel is {}'.format(self.parallel))

        print('ext.sourcedir is {}'.format(ext.sourcedir))

        print('***************************')

        cmake_args = "-DFASTASR_BUILD_PYTHON_MODULE=ON"
        cmake_args += f" -DCMAKE_INSTALL_PREFIX={Path(self.build_lib).resolve()}"

        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(self.build_lib, exist_ok=True)

        if is_windows():
            ret = os.system(
                f"cmake {cmake_args} -B {self.build_temp} -S {ext.sourcedir}"
            )
            if ret != 0:
                raise Exception("Failed to configure")

            ret = os.system(
                f"cmake --build {self.build_temp} --config Release"
            )
            if ret != 0:
                raise Exception("Failed to build fastasr")

            ret = os.system(
                f"cmake --install {self.build_temp} --config Release"
            )
            if ret != 0:
                raise Exception("Failed to install fastasr")
        else:
            ret = os.system(
                f"cd {self.build_temp};cmake {cmake_args} {ext.sourcedir};make -j8 install;pwd")


setup(
    name='fastasr',
    version='0.0.1',
    python_requires='>=3.6',
    install_requires=requirements,
    package_dir={"fastasr": "src/python/fastasr"},
    packages=["fastasr"],
    ext_modules=[CMakeExtension("cmake_example")],
    cmdclass={"build_ext": CMakeBuild},

)
