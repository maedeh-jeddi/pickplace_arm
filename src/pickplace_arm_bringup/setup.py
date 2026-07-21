import os
from glob import glob
from setuptools import setup

package_name = 'pickplace_arm_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.xml')),
        (os.path.join('share', package_name, 'maps'),
            glob('maps/*.yaml') + glob('maps/*.pgm')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maedeh Jeddi',
    maintainer_email='maedehjeddi1993@gmail.com',
    description='Pick and place automation for pickplace_arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pick_and_place = pickplace_arm_bringup.pick_and_place:main',
            'search_and_pick = pickplace_arm_bringup.search_and_pick:main',
            'nav_and_pick = pickplace_arm_bringup.nav_and_pick:main',
            'mission = pickplace_arm_bringup.mission:main',
            'mission_2 = pickplace_arm_bringup.mission_2:main',
            'teleop_key = pickplace_arm_bringup.teleop_key:main',
        ],
    },
)
