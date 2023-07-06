from setuptools import find_packages, setup

with open("basicNeuralNets/README.md", "r") as f:
    long_description = f.read()

setup(
    name="basicNeuralNetworks",
    version="0.0.1",
    description="Basic Neural Networks implemented from scratch.",
    package_dir={"": "basicNeuralNets"},
    packages=find_packages(where="basicNeuralNets"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Artshouldterrify/bNN",
    author="StubhLal",
    author_email="stubh.lal@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "pandas", "matplotlib", "random"],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.10",
)
