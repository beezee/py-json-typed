from setuptools import setup, find_packages
setup(
    name="py-json-typed",
    version="0.1",
    packages=find_packages(),
    author="beezee",
    description="Well typed json parsing",
    keywords="kafka",
    url="https://github.com/beezee/py-json-typed",
    install_requires=[
      'py-foldadt'
    ],
    project_urls={
        "Source Code": "https://github.com/beezee/py-json-typed",
    }
)
