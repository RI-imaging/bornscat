#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import dirname, realpath
from setuptools import setup, Command
import subprocess
import sys


author = u"Paul Müller"
authors = [author]
name = 'bornscat'
description = 'scattering code with Born/Rytov approximation'
year = "2015"

long_description = """
python library to compute forward scattering in the Born or Rytov approximations
"""

import bornscat
version=bornscat.__version__

class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='paul.mueller at biotec.tu-dresden.de',
        url='http://paulmueller.github.io/bornscat/',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=long_description,
        install_requires=["NumPy>=1.7.0", "SciPy>=0.10.0"],
        platforms=['ALL'],
        cmdclass = {'test': PyTest,
                    },
        )
