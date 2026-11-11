
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace fixtures {
// Exact float literals from the public 32-vector reproducer for issue #3290.
// https://github.com/user-attachments/files/34474982/repro.cpp
inline constexpr float FILTERED_BRIDGE_VECTORS[][4] = {
    {0.390625F, 0.453125F, 0.515625F, 0.578125F},
    {0.40625F, 0.46875F, 0.53125F, 0.59375F},
    {0.421875F, 0.484375F, 0.546875F, 0.609375F},
    {0.4375F, 0.5F, 0.5625F, 0.625F},
    {0.453125F, 0.515625F, 0.578125F, 0.640625F},
    {0.46875F, 0.53125F, 0.59375F, 0.65625F},
    {0.484375F, 0.546875F, 0.609375F, 0.671875F},
    {0.5F, 0.5625F, 0.625F, 0.6875F},
    {1.05F, 1.1125F, 1.175F, 1.2375F},
    {1.1F, 1.1625F, 1.225F, 1.2875F},
    {1.15F, 1.2125F, 1.275F, 1.3375F},
    {1.2F, 1.2625F, 1.325F, 1.3875F},
    {1.25F, 1.3125F, 1.375F, 1.4375F},
    {1.3F, 1.3625F, 1.425F, 1.4875F},
    {1.35F, 1.4125F, 1.475F, 1.5375F},
    {1.4F, 1.4625F, 1.525F, 1.5875F},
    {1.45F, 1.5125F, 1.575F, 1.6375F},
    {1.5F, 1.5625F, 1.625F, 1.6875F},
    {1.55F, 1.6125F, 1.675F, 1.7375F},
    {1.6F, 1.6625F, 1.725F, 1.7875F},
    {1.65F, 1.7125F, 1.775F, 1.8375F},
    {1.7F, 1.7625F, 1.825F, 1.8875F},
    {1.75F, 1.8125F, 1.875F, 1.9375F},
    {1.8F, 1.8625F, 1.925F, 1.9875F},
    {1.85F, 1.9125F, 1.975F, 2.0375F},
    {1.9F, 1.9625F, 2.025F, 2.0875F},
    {1.95F, 2.0125F, 2.075F, 2.1375F},
    {2.0F, 2.0625F, 2.125F, 2.1875F},
    {2.05F, 2.1125F, 2.175F, 2.2375F},
    {2.1F, 2.1625F, 2.225F, 2.2875F},
    {2.15F, 2.2125F, 2.275F, 2.3375F},
    {2.2F, 2.2625F, 2.325F, 2.3875F},
};
}  // namespace fixtures
