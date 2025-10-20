from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seal-calibration",
    version="0.1.0",
    description="Calibration library for 3DMakerPro SEAL 3D scanners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anton Sychev",
    url="https://github.com/klich3/3dMakerPro-SEAL-Calibration-script",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "viz": ["matplotlib>=3.4.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
