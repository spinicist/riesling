"""Setup file to Not Another Neuroimaging Slicer

"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, '../README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='riesling',
    version='0.1',
    description='Helper code for using riesling in Jupyter notebooks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/spinicist/riesling',
    author='Tobias Wood',
    author_email='tobias@spinicist.org.uk',
    py_modules=['riesling'],
    install_requires=['matplotlib>=3.2.0',
                      'h5py>=3.2.1',
                      'numpy>=1.14.2',
                      'colorcet>=2.0.0',
                      'cmasher',
                      'xarray'],
    python_requires='>=3',
    license='MPL',
    classifiers=['Topic :: Scientific/Engineering :: Visualization',
                 'Programming Language :: Python :: 3',
                 ],
    keywords='neuroimaging nifti',
    packages=find_packages(),
    entry_points={
        # 'console_scripts': [ ],
    },
)
