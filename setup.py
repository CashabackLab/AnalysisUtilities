from setuptools import setup
import re
import ast

#Only change __version__ in analysis_utilities/__init__.py file
_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('analysis_utilities/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))
setup(
    # Needed to silence warnings 
    name='AnalysisUtilities',
    url='https://github.com/CashabackLab/AnalysisUtilities',
    author='CashabackLab',
    author_email='cashabacklab@gmail.com',
    # Needed to actually package something
    packages=['analysis_utilities', 'analysis_utilities.roth_analysis', 'analysis_utilities.calalo_analysis','analysis_utilities.stats'],
    # Needed for dependencies
    install_requires=['numpy', 'numba > 0.54', 'scipy', 'tqdm'],
    # *strongly* suggested for sharing
    version=version,
    # The license can be anything you like
    license='MIT',
    description='Python package for analyzing human kinematic data, tailored for the Cashaback Lab',
    long_description=open('README.md').read(),
)
