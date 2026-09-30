from setuptools import setup, find_packages

# Read requirements.in, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.in", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except:
    print("'requirements.in' not found!")
    REQUIRES = list()

setup(
    name="rl",
    version="0.3.0",
    include_package_data=True,
    author="Bicheng Wang",
    url="https://github.com/BichengWang/RL-pricing",
    license="MIT",
    packages=find_packages(),
    #install_requires=REQUIRES,
    install_requires=REQUIRES
    + ["pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"],
    # dependency_links=['git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2'],
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
    keywords="Reinforcment Learning",
    platform=["any"],
    python_requires=">=3.6",
)
