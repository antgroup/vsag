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

"""LangChain's standard VectorStore integration tests for VSAGVectorStore."""

import pytest

pytest.importorskip("langchain_tests")

from langchain_tests.integration_tests.vectorstores import (  # noqa: E402
    VectorStoreIntegrationTests,
)

from pyvsag.langchain import VSAGVectorStore  # noqa: E402

pytestmark = pytest.mark.asyncio


class TestVSAGVectorStore(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        return False

    @pytest.fixture()
    def vectorstore(self):
        return VSAGVectorStore(self.get_embeddings())
