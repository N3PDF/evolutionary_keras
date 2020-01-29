from setuptools import setup, find_packages

setup(
        name="evolutionary_keras",
        version="1.0.0",
        author = 'S. Carrazza, J. Cruz-Martinez, Roy Stegeman',
        author_email = 'juan.cruz@mi.infn.it, roy.stegeman@mi.infn.it',
        url='https://github.com/N3PDF/evolutionary_keras',
        package_dir={'':'src'},
        packages=find_packages('src'),
        install_requires=[
            'numpy',
            'keras',
            'sphinx_rtd_theme',
            'recommonmark',
            ],
        python_requires='>=3.6'
        )
