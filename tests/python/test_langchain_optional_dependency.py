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

"""Checks that LangChain stays an optional dependency."""

import subprocess
import sys
import textwrap


def test_importing_pyvsag_does_not_import_langchain():
    code = "import sys, pyvsag; assert 'langchain_core' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_adapter_import_error_explains_the_extra():
    code = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "langchain_core" or fullname.startswith("langchain_core."):
                    raise ImportError("blocked for test")

        sys.meta_path.insert(0, Blocker())
        try:
            import pyvsag.langchain  # noqa: F401
        except ImportError as exc:
            assert 'pyvsag[langchain]' in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError")
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
