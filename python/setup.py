from setuptools import setup, Extension

# All other configuration is in setup.cfg / pyproject.toml.
# This file is kept for the C-extension definition.
setup(
    ext_modules=[Extension('example', sources=['example.c'])],
    zip_safe=False,
)
