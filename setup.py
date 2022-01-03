from setuptools import setup

setup(
    # Needed to silence warnings 
    name='AnalysisUtilities',
    url='https://github.com/CashabackLab/AnalysisUtilities',
    author='CashabackLab',
    author_email='cashabacklab@gmail.com',
    # Needed to actually package something
    packages=['statistics'],
    # Needed for dependencies
    install_requires=['numpy', 'numba', 'scipy', 'warnings'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='UD',
    description='Python package for analyzing human kinematic data, tailored for the Cashaback Lab',
    long_description=open('README.md').read(),
)
