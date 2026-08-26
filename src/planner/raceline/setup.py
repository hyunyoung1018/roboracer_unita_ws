import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'raceline'

setup(
    name=package_name,
    version='0.0.0',
    # find_packages also picks up the vendored TUM subtree, which is why
    # raceline/global_racetrajectory_optimization/__init__.py has to exist.
    # Its non-python data (params/*.ini, inputs/**) is deliberately NOT installed:
    # vehicle parameters come from stack_master/config/<mode>/ instead, so the
    # optimizer inputs live with the rest of the car config.
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.xml'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='young',
    maintainer_email='imsunghy@gmail.com',
    description='Global raceline generation and publishing for roboracer',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'raceline_generator = raceline.raceline_generator:main',
            'raceline_publisher = raceline.raceline_publisher:main',
            'raceline_tuner = raceline.raceline_tuner:main',
        ],
    },
)
