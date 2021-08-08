from setuptools import setup

setup(
    name='windroses',
    version='0.0.1',
    description='Plot wind roses',
    author='Marcia Marques',
    author_email='marcia.marques@alumni.usp.br',
    url='https://github.com/marcia-marques/wind-rose',
    packages=["windroses"],
    install_requires=[
        "numpy >= 1.21",
        "matplotlib >= 3.2",
        "pandas >= 1.3.1",
        "seaborn >= 0.11.1",
    ],
    )
