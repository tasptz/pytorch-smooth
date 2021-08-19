from setuptools import setup

with open('README.md') as f:
    description = f.read()

setup(
    name='smooth',
    version='0.1.0',
    description='Calculate the hessian matrix or smoothness value for feature maps ',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Thomas Pönitz',
    author_email='tasptz@gmail.com',
    url='https://github.com/tasptz/pytorch-smooth',
    packages=['smooth'],
    license='MIT',
    platforms=['any'],
    install_requires=['torch', 'numpy']
)