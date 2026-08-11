from setuptools import find_packages, setup


package_name = 'obstacle_router'

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
    description='Lossless static/dynamic obstacle stream router',
    license='MIT',
    entry_points={
        'console_scripts': [
            'obstacle_router_node = obstacle_router.obstacle_router_node:main',
        ],
    },
)
