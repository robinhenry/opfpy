from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='opfpy',
      version='0.0.1',
      url='http://github.com/robinhenry/opfpy',
      author='Robin Henry',
      description="A framework to model power systems in Python.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='robin.x.henry@gmail.com',
      packages=['opfpy'],
      install_requires=[],
      python_requires='>=3.5'
      )
