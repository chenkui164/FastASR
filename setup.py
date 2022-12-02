import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages
from pathlib import Path
import platform
import pylib_openblas as openblas
import pylib_fftw3f as fftw3f
import glob


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
        # print(' openblas.include_dirs is {}'.format(openblas.include_dirs[0]))
        # print(' openblas.library_dir is {}'.format(openblas.library_dir))
        openblas_home = Path(openblas.library_dir).parent.absolute()
        openblas_lib = glob.glob(os.path.join(openblas_home, 'lib*'))[0]
        openblas_bin = os.path.join(openblas_home, 'bin')
        openblas_bin = openblas_bin.replace("\\", "\\\\")
        print((openblas_bin))
        print('openblas_lib is {}'.format(openblas_lib))
        print('openblas_home is {}'.format(openblas_home))
        openblas_include = openblas.include_dir
        fftw3f_include = fftw3f.include_dir[0]
        fftw3f_lib = fftw3f.library_dir

        ret = os.system(
            f"dir {openblas_lib}"
        )

        ret = os.system(
            f"dir {fftw3f_lib}"
        )
        print('***************************')

        cmake_args = "-DFASTASR_BUILD_PYTHON_MODULE=ON"
        cmake_args += f" -DCMAKE_INSTALL_PREFIX={Path(self.build_lib).resolve()}"
        cmake_args += f" -DOPENBLAS_INCLUDE_DIR={openblas_include}"
        cmake_args += f" -DOPENBLAS_LIBRARY_DIR={openblas_lib}"
        cmake_args += f" -DOPENBLAS_BIN_DIR={openblas_bin}"
        cmake_args += f" -DFFTW3F_INCLUDE_DIR={fftw3f_include}"
        cmake_args += f" -DFFTW3F_LIBRARY_DIR={fftw3f_lib}"

        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(self.build_lib, exist_ok=True)

        if is_windows():
            ret = os.system(
                f"cmake {cmake_args} -A {Win32} -B {self.build_temp} -S {ext.sourcedir}"
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
            ret = os.system("echo --------------------------------")

            self.env = os.popen('which python').read()
            self.env = os.path.dirname(self.env)
            self.env = os.path.dirname(self.env)
            print('self.env is {}'.format(self.env))
            ret = os.system("echo $PATH")
            ret = os.system("echo --------------------------------")
            ret = os.system(
                f"export VIRTUAL_ENV={self.env};cd {self.build_temp};cmake {cmake_args} -DVIRTUAL_ENV={self.env} {ext.sourcedir};make -j8 install;pwd")


setup(
    name='fastasr',
    version='0.0.2',
    python_requires='>=3.6',
    install_requires=requirements,
    package_dir={"fastasr": "src/python/fastasr"},
    packages=["fastasr"],
    ext_modules=[CMakeExtension("cmake_example")],
    cmdclass={"build_ext": CMakeBuild},

)
