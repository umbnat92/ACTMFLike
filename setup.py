from setuptools import find_packages, setup

setup(
    name="actmflike",
    version="1.0",
    description="Multi-frequency ACT likelihood for cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "cobaya>=3.0",
        "sacc>=0.4.2",
    ],
    package_data={"actmflike": ["MFLikeACT.yaml"]},
)
