from __future__ import absolute_import, division, print_function
import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)

    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        default_path = os.path.join(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path(
            'nvcc', os.environ['PATH'] + os.pathsep + default_path)

        if nvcc is None:
            raise EnvironmentError(
                'The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME')

        home = os.path.dirname(os.path.dirname(nvcc))

    cuda_config = {
        'home': home,
        'nvcc': nvcc,
        'include': os.path.join(home, 'include'),
        'lib64': os.path.join(home, 'lib64')
    }

    for k, v in cuda_config.items():
        if not os.path.exists(v):  # not found
            raise EnvironmentError(
                'The CUDA %s path could not be located in %s' % (k, v))

    return cuda_config


# C__numpy
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# if not using cuda, comment out this line and nvcc_compiler
CUDA = locate_cuda()


def nvcc_compiler(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    # compile .cu
    self.src_extensions.append('.cu')
    # save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)

        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler_so = default_compiler_so

    # inject redefined _compile method
    self._compile = _compile


class proj_build_ext(build_ext):
    def build_extensions(self):
        # if not using cuda, comment out this line
        nvcc_compiler(self.compiler)

        build_ext.build_extensions(self)


# if error at "{'gcc': ["-Wno-cpp", "-Wno-unused-function"]}" when not using cuda,
# change this line to "["-Wno-cpp", "-Wno-unused-function"]"
ext_modules = [
    Extension(
        'bbox.transform',
        ['bbox/transform.pyx'],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),

    Extension(
        'bbox.overlaps',
        ['bbox/overlaps.pyx'],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),

    Extension(
        'nms.cpu_nms',
        ['nms/cpu_nms.pyx'],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),

    # if not using cuda, comment out this extension
    Extension(
        'nms.gpu_nms',
        ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': ["-Wno-unused-function"],
            'nvcc': ['-arch=sm_61',
                     '--ptxas-options=-v',
                     '-c',
                     '--compiler-options',
                     "'-fPIC'"]
        },
        include_dirs=[numpy_include, CUDA['include']]
    )
]

setup(
    name='tf-faster-rcnn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': proj_build_ext}
)
