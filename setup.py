import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="candle_hayes",  ### Cannot be same name as package ?
    version="1.0.7",
    author="Hayes",
    author_email="hayfluss@gmail.com",
    description="Toolkit for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThayaFluss/candle",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
