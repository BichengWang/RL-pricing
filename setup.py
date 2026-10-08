from setuptools import setup, find_packages

# Read requirements.in, ignore comments
REQUIRES = []
try:
    with open("requirements.in", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.in' not found!")

setup(
    name="rl",
    version="0.4.0",
    include_package_data=True,
    author="Bicheng Wang",
    url="https://github.com/BichengWang/RL-pricing",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIRES,
    extras_require={"test": ["pytest>=7.0"], "lint": ["ruff>=0.5"]},
    entry_points={"console_scripts": ["rl-pricing=rl.cli:main"]},
    description="Deep reinforcement learning for stock portfolio trading, adapted from the FinRL library.",
    long_description="""RL-pricing applies deep reinforcement learning (A2C, PPO, DDPG, SAC,
    TD3 and a rolling ensemble) to multi-stock portfolio trading and benchmarks
    the results against rule-based portfolios and market indices.

    The environment, preprocessing, and training code are adapted from FinRL
    (https://github.com/AI4Finance-Foundation/FinRL), which is MIT-licensed.
    """,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="reinforcement learning, stock trading",
    platforms=["any"],
    python_requires=">=3.10",
)
