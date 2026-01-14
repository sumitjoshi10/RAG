from setuptools import find_packages,setup

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"


REPO_NAME = "RAG"
AUTHOR_USER_NAME = "sumitjoshi10"
SRC_REPO = "RAG"
AUTHOR_EMAIL = "sumit.joshi9818@mgmail.com"

setup(
    name= SRC_REPO,
    version= __version__,
    author= AUTHOR_USER_NAME,
    author_email= AUTHOR_EMAIL,
    description="A python package for RAG - Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(where = "src")
)