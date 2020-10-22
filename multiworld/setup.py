from distutils.core import setup
from setuptools import find_packages
 
setup(
    name='multiworld',
    packages=find_packages(),
    package_data={'multiworld': [
        'envs/assets/classic_mujoco/*.xml',
        ]
    },
)
