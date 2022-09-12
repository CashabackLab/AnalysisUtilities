from setuptools import setup

setup(
    # Needed to silence warnings 
    name='AnalysisUtilities',
    url='https://github.com/CashabackLab/AnalysisUtilities',
    author='CashabackLab',
    author_email='cashabacklab@gmail.com',
    # Needed to actually package something
    packages=['analysis_utilities', 'analysis_utilities.roth_analysis', 'analysis_utilities.stats'],
    # Needed for dependencies
    install_requires=['numpy', 'numba', 'scipy', 'tqdm'],
    # *strongly* suggested for sharing
    version='0.4.10',
    # The license can be anything you like
    license='MIT',
    description='Python package for analyzing human kinematic data, tailored for the Cashaback Lab',
    long_description=open('README.md').read(),
)
