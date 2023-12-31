#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ "ipywidgets==7.6.5", "lightkurve==2.0.11", "matplotlib==3.3.4", "pandas==1.1.5", "scipy==1.5.3",]

test_requirements = [ ]

setup(
    author="Daniel Giles",
    author_email='daniel.k.giles@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package for phase folding tools. Developed for use with TESS light curves on Eclipsing Binary candidates",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='phasefolder',
    name='phasefolder',
    packages=find_packages(include=['phasefolder', 'phasefolder.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/d-giles/phasefolder',
    version='0.3.2',
    zip_safe=False,
)
