from setuptools import setup

setup(
    name='sppa',
    version='1.0.5',
    packages=['sppa'],
    install_requires=['numpy', 'pulp'],
    url='https://github.com/jamespltan/sppa',
	download_url='https://github.com/jamespltan/sppa/archive/v1.0.5.tar.gz',
    license='MIT',
    author='James Tan',
    author_email='jamestan@ntu.edu.sg',
    description='SPPA MINLP solver'
)
