#!/usr/bin/python
# Copyright (c) 2009 Las Cumbres Observatory.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from setuptools import setup, find_packages


DESCRIPTION = """Common libraries for the conversion and scaling of fits images"""


setup(
    name="fits2image",
    version="0.5.0",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author='Jon Nation',
    author_email='jnation@lcogt.net',
    packages=find_packages('.', exclude=['tests', 'tests.*']),
    # 3.10 is a hard floor: conversions.py uses match statements.
    python_requires='>=3.10',
    # Floors are the oldest combination the test suite is known to pass on. There are
    # deliberately no upper bounds - the range is proved by CI rather than guessed at here.
    install_requires=['numpy>=1.22', 'astropy>=5.0', 'Pillow>=8.4']
)
