"""
Setup script for the inverse-canopy package
"""
from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='orderdd',
    url='https://openpra.org',
    version='0.0.1',
    author='Arjun Earthperson',
    author_email='arjun@openpra.org',
    license='AGPL-3.0',
    description='Tensorflow based library predicting optimal variable orderings for BDDs.',
    long_description_content_type='text/markdown',
    long_description=README,
    install_requires=[
        'collections',
        'dd',
        'itertools',
        'matplotlib',
        'numpy'
        'pandas',
        'sklearn',
        'tensorflow'
    ],
    extras_require={
        'cuda': [
            'tensorflow[and-cuda]'
        ],
        'metal': [
            'tensorflow==2.13.1',
            'tensorflow-metal==1.0.0',
        ],
        'dev': [
            'pyarrow',
            'pytest',
            'pytest-cov',
            'pylint',
            'pylint[spelling]',
            'twine'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed'
    ],
    packages=['orderdd'],
    python_requires='>=3.8',
    entry_points={
    },
)
