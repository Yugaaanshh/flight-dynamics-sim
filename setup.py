"""
Flight Dynamics Simulator - Setup Script

Install in development mode:
    pip install -e .

Then import anywhere:
    from flight_dynamics_sim.config_loader import load_aircraft
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        install_requires = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    install_requires = [
        'numpy>=1.24',
        'scipy>=1.10',
        'matplotlib>=3.7',
        'streamlit>=1.28',
        'plotly>=5.15',
        'pandas>=2.0',
        'pyyaml>=6.0',
    ]

setup(
    name="flight-dynamics-sim",
    version="1.0.0",
    author="Flight Dynamics Team",
    description="Complete 6-DOF aircraft flight dynamics simulator based on Roskam",
    long_description=open("README.md", encoding="utf-8").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Yugaanshh/flight-dynamics-sim",
    # Map the current directory as the flight_dynamics_sim package
    package_dir={'flight_dynamics_sim': '.'},
    packages=['flight_dynamics_sim', 'flight_dynamics_sim.eom', 'flight_dynamics_sim.sim', 
              'flight_dynamics_sim.aero', 'flight_dynamics_sim.analysis', 
              'flight_dynamics_sim.control', 'flight_dynamics_sim.experiments'],
    package_data={
        '': ['*.yaml', '*.yml'],
    },
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "fds-ui=streamlit_app:main",
        ],
    },
)
