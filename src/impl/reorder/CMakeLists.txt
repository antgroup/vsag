
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


set (REORDER_SRC
        pqr_reorder.cpp
        pqr_reorder.h
        flatten_reorder.cpp
        flatten_reorder.h
        reorder.h
        reorder_parameter.h
        reorder_parameter.cpp
)

add_library (reorder OBJECT ${REORDER_SRC})
