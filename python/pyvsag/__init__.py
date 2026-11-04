
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

_cur_file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_cur_file_dir)

# The native module emits its initialization banner while it is being loaded. Suppress it before
# that import so API consumers and the JSON CLI do not get unexpected stdout output.
os.environ.setdefault("VSAG_SUPPRESS_INIT_BANNER", "1")

from ._index import Index, _check
from ._lib import lib as _lib
from ._version import __version__
from .analyzer import ANALYSIS_SCHEMA_VERSION, analyze_index

_LOG_LEVELS = {"off": _lib.VSAG_LOG_OFF, "info": _lib.VSAG_LOG_INFO, "debug": _lib.VSAG_LOG_DEBUG}


def _set_log_level(name):
    _check(_lib.vsag_set_log_level(_LOG_LEVELS[name]), "set log level failed")


def set_logger_off():
    """Disable all logging output from the VSAG library."""
    _set_log_level("off")


def set_logger_info():
    """Set logger level to INFO. Only important information will be logged."""
    _set_log_level("info")


def set_logger_debug():
    """Set logger level to DEBUG. Detailed debug information will be logged."""
    _set_log_level("debug")


__all__ = ["Index", "set_logger_off", "set_logger_info", "set_logger_debug", "__version__"]

