import sys
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

try:
    from torch.utils.cpp_extension import CUDAExtension
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False

if sys.platform == "win32":
    cxx_args = ["/O2", "/openmp"]
    link_args = ["/openmp"]
    nvcc_args = ["-O3", "--use_fast_math"]
else:
    cxx_args = ["-O3", "-fopenmp", "-march=native"]
    link_args = ["-fopenmp"]
    nvcc_args = ["-O3", "--use_fast_math"]

ext_modules = [
    CppExtension(
        "top_coactivation_reduce",
        ["top_coactivation_reduce.cpp"],
        extra_compile_args={"cxx": cxx_args},
        extra_link_args=link_args,
    ),
    CppExtension(
        "mid_reservoir",
        ["mid_reservoir.cpp"],
        extra_compile_args={"cxx": cxx_args},
        extra_link_args=link_args,
    ),
]

if _CUDA_AVAILABLE:
    ext_modules.append(
        CUDAExtension(
            "latent_stats_cuda",
            ["latent_stats_cuda.cu"],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    )
    ext_modules.append(
        CUDAExtension(
            "linear_relu_ext",
            ["linear_relu.cu"],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
            extra_link_args=["-lcublasLt"],
        )
    )
else:
    print("WARNING: CUDA not available — latent_stats_cuda and linear_relu_ext will not be built.")

setup(
    name="native_extensions",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
