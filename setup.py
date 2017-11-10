# from distutils.core import setup
from setuptools import setup

setup(
    name='gp_project',
    # packages=['gp_project'],
    py_modules=['gp_project'],
    version='0.1',
    description='A Gaussian Process Project for STAT8810 at OSU',
    author='Jordan Melendez',
    author_email='jmelendez1992@gmail.com',
    license='MIT',
    url='https://github.com/jordan-melendez/gp_project',
    # download_url='',
    test_suite='nose.collector',
    tests_require=['nose'],
    keywords='EFT nuclear model gaussian process uncertainty quantification',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
        ]
)
