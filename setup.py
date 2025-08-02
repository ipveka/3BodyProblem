#!/usr/bin/env python3
"""
Setup script for 3BodyProblem N-Body Gravitational Simulation package.

This script enables installation of the 3BodyProblem package and its dependencies.
It supports both development and production installations.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we're using Python 3.7+
if sys.version_info < (3, 7):
    sys.exit('Python 3.7 or later is required.')

# Read the README file for long description
def read_readme():
    """Read README.md file for long description."""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "N-Body Gravitational Simulation with Streamlit Interface"

# Read version from a version file or set default
def get_version():
    """Get package version."""
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return '1.0.0'

# Define package requirements
REQUIREMENTS = [
    # Core scientific computing
    'numpy>=1.19.0',
    'scipy>=1.7.0',
    
    # Visualization
    'matplotlib>=3.3.0',
    'plotly>=5.0.0',
    'seaborn>=0.11.0',
    
    # Web interface
    'streamlit>=1.25.0',
    
    # Utilities
    'pandas>=1.3.0',  # For data handling in Streamlit
]

# Development requirements (optional)
DEV_REQUIREMENTS = [
    'pytest>=6.0.0',
    'pytest-cov>=2.10.0',
    'black>=21.0.0',
    'flake8>=3.8.0',
    'mypy>=0.910',
    'jupyter>=1.0.0',
    'notebook>=6.0.0',
]

# Documentation requirements (optional)
DOC_REQUIREMENTS = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'myst-parser>=0.15.0',
]

# Performance requirements (optional)
PERFORMANCE_REQUIREMENTS = [
    'numba>=0.56.0',  # For JIT compilation acceleration
    'cython>=0.29.0',  # For compiled extensions
]

setup(
    # Package metadata
    name='3BodyProblem',
    version=get_version(),
    author='N-Body Simulation Team',
    author_email='contact@3bodyproblem.dev',
    description='Comprehensive N-Body Gravitational Simulation with Interactive Visualization',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/3BodyProblem',
    project_urls={
        'Documentation': 'https://3bodyproblem.readthedocs.io/',
        'Source': 'https://github.com/your-username/3BodyProblem',
        'Tracker': 'https://github.com/your-username/3BodyProblem/issues',
    },
    
    # Package discovery
    packages=find_packages(exclude=['tests*', 'docs*']),
    include_package_data=True,
    
    # Requirements
    python_requires='>=3.7',
    install_requires=REQUIREMENTS,
    extras_require={
        'dev': DEV_REQUIREMENTS,
        'docs': DOC_REQUIREMENTS,
        'performance': PERFORMANCE_REQUIREMENTS,
        'all': DEV_REQUIREMENTS + DOC_REQUIREMENTS + PERFORMANCE_REQUIREMENTS,
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            '3body-demo-two=scripts.run_two_body:main',
            '3body-demo-three=scripts.run_three_body:main',
            '3body-streamlit=app.streamlit_app:main',
        ],
    },
    
    # Package classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Framework :: Matplotlib',
        'Framework :: Jupyter',
    ],
    
    # Keywords for package discovery
    keywords=[
        'physics', 'simulation', 'gravity', 'orbital-mechanics', 
        'n-body', 'celestial-mechanics', 'astronomy', 'visualization',
        'streamlit', 'interactive', 'educational', 'scientific-computing'
    ],
    
    # Package data
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml', '*.toml'],
        'app': ['*.py'],
        'core': ['*.py'],
        'visualization': ['*.py'],
        'scripts': ['*.py'],
    },
    
    # Data files (external to package)
    data_files=[
        ('config', ['.streamlit/config.toml']),
    ],
    
    # Zip safety
    zip_safe=False,
    
    # License
    license='MIT',
    
    # Additional metadata
    platforms=['any'],
    
    # Custom commands can be added here
    cmdclass={},
    
    # Obsoletes and provides (for package replacement)
    # obsoletes=[],
    # provides=[],
)

# Post-installation message
def print_installation_message():
    """Print installation success message with usage instructions."""
    print("\n" + "="*60)
    print("🌌 3BodyProblem Installation Complete! 🌌")
    print("="*60)
    print("\nQuick Start:")
    print("1. Run two-body demo:     3body-demo-two")
    print("2. Run three-body demo:   3body-demo-three")
    print("3. Launch Streamlit app:  streamlit run app/streamlit_app.py")
    print("\nOr import in Python:")
    print("  from core.simulation import NBodySimulation")
    print("  from core.body import CelestialBody")
    print("\nFor more information, see README.md")
    print("="*60)

# Custom install command to show message
from setuptools.command.install import install
from setuptools.command.develop import develop

class PostInstallCommand(install):
    """Custom install command to show post-installation message."""
    def run(self):
        install.run(self)
        print_installation_message()

class PostDevelopCommand(develop):
    """Custom develop command to show post-installation message."""
    def run(self):
        develop.run(self)
        print_installation_message()

# Update setup with custom commands
setup.cmdclass = {
    'install': PostInstallCommand,
    'develop': PostDevelopCommand,
}

# Version compatibility checks
def check_dependencies():
    """Check if critical dependencies are available."""
    try:
        import numpy
        import scipy
        import matplotlib
        import streamlit
        print("✓ All critical dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing critical dependency: {e}")
        return False

if __name__ == '__main__':
    # Run dependency check during direct execution
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'develop']:
        print("Checking dependencies...")
        check_dependencies()
