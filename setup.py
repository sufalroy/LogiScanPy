import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="logiscanpy",
    version="0.1.0",
    author="Sufal Roy",
    author_email="sufalroy1997@google.com",
    description="Real-time object counting application for conveyor belts using YOLOv8 and OpenCV.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sufalroy/LogiScanPy.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
