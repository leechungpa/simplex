# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.methods.simclr import SimCLR
from solo.methods.simplex import Simplex
from solo.methods.dcl import DCL
from solo.methods.dhel import DHEL


METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "simclr": SimCLR,
    "simplex": Simplex,
    "dcl": DCL,
    "dhel": DHEL,
}
__all__ = [
    "BaseMethod",
    "LinearModel",
    "SimCLR",
    "simplex",
    "dcl",
    "dhel",
]
