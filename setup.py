import setuptools
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name='sdm_assignment_1',
    version='1.0',
    description='A Scientific Data Managemnet Assignment',
    url='https://github.com/samhiggs/sdm_kmeans',
    classifiers=['Programming Language :: Python :: 3.7'],
    py_modules=["kmeans"],
    python_requires='>=3.0',
)