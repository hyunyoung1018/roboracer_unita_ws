from setuptools import setup

package_name = 'parameter_event_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='young',
    maintainer_email='imsunghy@gmail.com',
    description="rclpy ParameterEventHandler, backported for Humble",
    license='Apache License 2.0',
    entry_points={'console_scripts': []},
)
