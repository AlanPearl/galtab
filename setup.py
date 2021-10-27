from setuptools import setup, find_packages

setup(name="hodtab",
      version="0.0.1.dev1",
      description="A general approach to tabulating HOD statistics",
      url="http://github.com/AlanPearl/hodtab",
      author="Alan Pearl",
      author_email="alanpearl@pitt.edu",
      license="MIT",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
            "numpy",
            "jax",
            "halotools>=0.7",
      ],
      zip_safe=True,
      test_suite="nose.collector",
      tests_require=["nose"],
      )
