from setuptools import find_packages, setup

package_name = 'spline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unita',
    maintainer_email='unita@todo.todo',
    description='Frenet spline obstacle avoidance for roboracer_unita_ws',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    # h2h_spline_node is the head-to-head subclass. It is a separate executable
    # so that time_trials keeps running the unmodified spline_node.
    entry_points={'console_scripts': [
        'spline_node = spline.spline_node:main',
        'h2h_spline_node = spline.h2h_spline_node:main',
    ]},
)
