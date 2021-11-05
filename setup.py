from setuptools import setup, find_packages
# from distutils.extension import Extension
# try:
#     from Cython.Distutils import build_ext
#     import numpy as np
# except ImportError as err:
#     np = build_ext = None
#     build_kwargs = {}
#     raise
# else:
#     cic_ext = Extension(name="galtab.counts_in_cylinders.engines."
#                              "counts_in_cylinders_engine",
#                         sources=["galtab/counts_in_cylinders/engines/"
#                                  "counts_in_cylinders_engine.pyx"],
#                         language="c++",
#                         extra_compile_args=["-Ofast"],
#                         include_dirs=[np.get_include()])
#     build_kwargs = dict(ext_modules=[cic_ext],
#                         cmdclass={"build_ext": build_ext})
build_kwargs = {}

setup(name="galtab",
      version="0.0.1.dev1",
      description="A general approach to tabulating HOD statistics",
      url="https://github.com/AlanPearl/galtab",
      author="Alan Pearl",
      author_email="alanpearl@pitt.edu",
      license="MIT",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
          # conda install gxx_linux-64
          # "Cython",
          "numpy",
          "scipy",
          "jax",
          "halotools>=0.7",
      ],
      zip_safe=True,
      test_suite="nose.collector",
      tests_require=["nose"],
      **build_kwargs
      )
