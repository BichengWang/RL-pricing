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
    version="0.3.0",
    include_package_data=True,
    author="Bicheng Wang",
    url="https://github.com/BichengWang/RL-pricing",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRES
    + ["pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"],
    description="Deep reinforcement learning for stock portfolio trading, adapted from the FinRL library.",
    long_description="""RL-pricing applies deep reinforcement learning (A2C, PPO, DDPG, SAC,
    TD3) to multi-stock portfolio trading and benchmarks the results against SPY.

    The environment, preprocessing, and training code are adapted from FinRL
    (https://github.com/AI4Finance-Foundation/FinRL), which is MIT-licensed.
    """,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="reinforcement learning, stock trading",
    platforms=["any"],
    python_requires=">=3.6",
)
