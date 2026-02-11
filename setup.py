# setup.py

import subprocess
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _nvcc_cuda_version():
    """Detect CUDA version from nvcc to filter unsupported architectures."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        m = re.search(r"release (\d+)\.(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else (12, 0)
    except Exception:
        return (12, 0)


# Auto-detect supported GPU archs based on CUDA toolkit version
# 80: A100 (Ampere), 90: H100 (Hopper, CUDA 11.8+), 120: B100/B200 (Blackwell, CUDA 12.8+)
_cuda_ver = _nvcc_cuda_version()
supported_archs = ["80"]
if _cuda_ver >= (11, 8):
    supported_archs.append("90")
if _cuda_ver >= (12, 8):
    supported_archs.append("120")

cc_flag = []
for arch in supported_archs:
    cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    
# setup(
#     name='vecadd_extension',
#     ext_modules=[
#         CUDAExtension(
#             'vecadd_extension', # 模块名称
#             [
#                 'vecadd_kernel.cu',  # CUDA Kernel 文件
#                 'binding.cc', # C++ 绑定文件
#             ],
#             extra_compile_args={'nvcc': ['-O3'] + cc_flag} # 优化级别
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

setup(
    name='sparse_kernel_extension',
    ext_modules=[
        CUDAExtension(
            'sparse_kernel_extension',
            sources=['get_table_kernel.cu'],
            extra_compile_args={'nvcc': ['-O3'] + cc_flag}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
