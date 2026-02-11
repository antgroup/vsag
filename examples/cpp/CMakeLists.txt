
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


add_executable(101_index_hnsw 101_index_hnsw.cpp)
target_link_libraries(101_index_hnsw vsag)

add_executable(102_index_diskann 102_index_diskann.cpp)
target_link_libraries(102_index_diskann vsag_static)

add_executable (103_index_hgraph 103_index_hgraph.cpp)
target_link_libraries (103_index_hgraph vsag)

add_executable(104_index_fresh_hnsw 104_index_fresh_hnsw.cpp)
target_link_libraries(104_index_fresh_hnsw vsag)

add_executable (105_index_brute_force 105_index_brute_force.cpp)
target_link_libraries (105_index_brute_force vsag)

add_executable (106_index_ivf 106_index_ivf.cpp)
target_link_libraries (106_index_ivf vsag)

add_executable (107_index_pyramid 107_index_pyramid.cpp)
target_link_libraries (107_index_pyramid vsag)

add_executable (108_index_gno_imi 108_index_gno_imi.cpp)
target_link_libraries (108_index_gno_imi vsag)

add_executable (109_index_sindi 109_index_sindi.cpp)
target_link_libraries (109_index_sindi vsag)

add_executable (201_custom_allocator 201_custom_allocator.cpp)
target_link_libraries (201_custom_allocator vsag)

add_executable (202_custom_logger 202_custom_logger.cpp)
target_link_libraries (202_custom_logger vsag)

add_executable (203_custom_thread_pool 203_custom_thread_pool.cpp)
target_link_libraries (203_custom_thread_pool vsag)

add_executable (301_feature_filter 301_feature_filter.cpp)
target_link_libraries (301_feature_filter vsag)

add_executable(302_feature_range_search 302_feature_range_search.cpp)
target_link_libraries(302_feature_range_search vsag)

add_executable (303_feature_remove 303_feature_remove.cpp)
target_link_libraries (303_feature_remove vsag)

add_executable(304_feature_enhance_graph 304_feature_enhance_graph.cpp)
target_link_libraries(304_feature_enhance_graph vsag)

# add_executable(305_feature_update 305_feature_update.cpp)
# target_link_libraries(305_feature_update vsag)

add_executable(306_feature_calculate_distance_by_id 306_feature_calculate_distance_by_id.cpp)
target_link_libraries(306_feature_calculate_distance_by_id vsag)

add_executable(307_feature_check_features 307_feature_check_features.cpp)
target_link_libraries(307_feature_check_features vsag)

add_executable(308_feature_estimate_memory 308_feature_estimate_memory.cpp)
target_link_libraries(308_feature_estimate_memory vsag)

add_executable(309_feature_clone 309_feature_clone.cpp)
target_link_libraries(309_feature_clone vsag)

add_executable(310_feature_export_model 310_feature_export_model.cpp)
target_link_libraries(310_feature_export_model vsag)

add_executable(311_feature_train 311_feature_train.cpp)
target_link_libraries(311_feature_train vsag)

add_executable(312_feature_odescent 312_feature_odescent.cpp)
target_link_libraries(312_feature_odescent vsag)

 add_executable(313_feature_search_allocator 313_feature_search_allocator.cpp)
 target_link_libraries(313_feature_search_allocator vsag)

add_executable(314_feature_hgraph_search_allocator 314_feature_hgraph_search_allocator.cpp)
target_link_libraries(314_feature_hgraph_search_allocator vsag)

add_executable(315_feature_hgraph_merge 315_feature_hgraph_merge.cpp)
target_link_libraries(315_feature_hgraph_merge vsag)

add_executable (316_index_int8_hgraph 316_index_int8_hgraph.cpp)
target_link_libraries (316_index_int8_hgraph vsag)

add_executable(317_feature_get_detail_data 317_feature_get_detail_data.cpp)
target_link_libraries(317_feature_get_detail_data vsag)

add_executable(318_feature_tune 318_feature_tune.cpp)
target_link_libraries(318_feature_tune vsag)

add_executable(319_feature_get_memory_usage 319_feature_get_memory_usage.cpp)
target_link_libraries(319_feature_get_memory_usage vsag)

add_executable(320_feature_extra_info 320_feature_extra_info.cpp)
target_link_libraries(320_feature_extra_info vsag)

add_executable (401_persistent_kv 401_persistent_kv.cpp)
target_link_libraries (401_persistent_kv vsag)

add_executable (402_persistent_streaming 402_persistent_streaming.cpp)
target_link_libraries (402_persistent_streaming vsag)

add_executable (501_quantization_transform 501_quantization_transform.cpp)
target_link_libraries (501_quantization_transform vsag)
