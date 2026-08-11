from glob import glob
from setuptools import find_packages, setup


package_name = 'gp_traj_predictor'

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
    description='UNICORN-derived GP opponent trajectory predictor',
    license='MIT',
    entry_points={
        'console_scripts': [
            'opponent_trajectory = gp_traj_predictor.opponent_trajectory:main',
            'gaussian_process_opp_traj = gp_traj_predictor.gaussian_process_opp_traj:main',
            'opp_prediction = gp_traj_predictor.opp_prediction:main',
        ],
    },
)
