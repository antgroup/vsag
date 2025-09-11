from setuptools import setup, Extension

# DON'T REMOVE: to make the wheel's name contains python version
example_module = Extension('example', sources=['example.c'])

setup(
    ext_modules=[example_module],
)