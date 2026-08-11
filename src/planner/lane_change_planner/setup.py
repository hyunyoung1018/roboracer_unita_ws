from glob import glob
from setuptools import find_packages, setup


package_name = 'lane_change_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unita',
    maintainer_email='unita@todo.todo',
    description='UNICORN-derived lane-change obstacle planner',
    license='MIT',
    entry_points={
        'console_scripts': [
            'change_avoidance_node = lane_change_planner.change_avoidance_node:main',
        ],
    },
)
