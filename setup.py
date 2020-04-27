# setup.py
import setuptools

setuptools.setup(
    name="asteroestimate",
    version="0.0.1",
    author="J. Ted Mackereth",
    license="MIT",
    url="https://github.com/jmackereth/asteroestimate",
    author_email="tedmackereth@gmail.com",
    description="Python based estimation for asteroseismology",
    packages=setuptools.find_packages(include=['asteroestimate','asteroestimate.*'])
    )
