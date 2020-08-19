import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hayes@thayafluss", # Replace with your own username
    version="1.0.1",
    author="Hayes",
    author_email="hayfluss@gmail.com",
    description="Toolkit for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThayaFluss/candle",
    packageds = setuptools.find_packages(include=["candle", "candle.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
