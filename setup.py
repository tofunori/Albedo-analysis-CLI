from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="albedo-analysis-framework",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reproducible Python framework for analyzing MODIS albedo products against AWS data on glaciers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/albedo-analysis-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "albedo-analysis=main:main",
        ],
    },
    package_data={
        "": ["config/*.yaml"],
    },
    include_package_data=True,
)