from setuptools import find_packages, setup

package_name = 'spline_planner'

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
    description='Multi-obstacle dynamic spline planner for roboracer_unita_ws',
    license='MIT',
    entry_points={'console_scripts': [
        'dynamic_avoidance_node = spline_planner.dynamic_avoidance_node:main',
        'update_waypoints = spline_planner.update_waypoints:main',
        'recovery_node = spline_planner.recovery_node:main',
    ]},
)
