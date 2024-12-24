
import os
import sys

_cur_file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_cur_file_dir)

from _pyvsag import *
from ._version import __version__
