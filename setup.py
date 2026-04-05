from setuptools import setup, find_packages

setup(
    name="skinlesions",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "matplotlib>=3.6",
        "Pillow>=9.0",
        "tqdm>=4.64",
        "PyYAML>=6.0",
        "torch>=2.0",
        "torchvision>=0.15",
    ],
    entry_points={
        "console_scripts": [
            "skinlesions-make-splits=skinlesions.scripts.make_splits:main",
            "skinlesions-train=skinlesions.scripts.train:main",
            "skinlesions-evaluate=skinlesions.scripts.evaluate:main",
        ]
    },
)
