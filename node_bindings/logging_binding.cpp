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

#include <napi.h>

#include "binding.h"
#include "vsag/vsag.h"

namespace {

Napi::Value
SetLoggerOff(const Napi::CallbackInfo& info) {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kOFF);
    return info.Env().Undefined();
}

Napi::Value
SetLoggerInfo(const Napi::CallbackInfo& info) {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
    return info.Env().Undefined();
}

Napi::Value
SetLoggerDebug(const Napi::CallbackInfo& info) {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
    return info.Env().Undefined();
}

}  // namespace

void
InitLogging(Napi::Env env, Napi::Object exports) {
    exports.Set("setLoggerOff", Napi::Function::New(env, SetLoggerOff));
    exports.Set("setLoggerInfo", Napi::Function::New(env, SetLoggerInfo));
    exports.Set("setLoggerDebug", Napi::Function::New(env, SetLoggerDebug));
}
