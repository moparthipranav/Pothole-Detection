from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will read the requirements in filepath and and return the list of requirements
    
    :param file_path: The path of the file requirements.txt
    :type file_path: str
    :return: List of all the requirements which are required for installation
    :rtype: List[str]
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n ") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements
    
setup(
name = 'pothole-detection',
version='0.0.1',
author='Pranav',
author_email='moparthipranav5@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)