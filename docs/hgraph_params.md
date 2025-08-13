The document introduces how to modify the `factory json` for HGraph Index.

Different capabilities of HGraph Index can be enabled by configuring various parameter combinations.

You can use `vsag::Factory::CreateIndex("hgraph", hgraph_json_string)` to factory HGraph Index.
If necessary, you can use `vsag::Engine` with resource manager to create index too.

The following will introduce how to edit the `hgraph_json_string` like this

### Parameters

The example of the `hgraph_json_string`.

```json5
{
  "dtype": "float32", // data_type: only support float32 for hgraph
  "metric_type": "l2", // metric_type only support "l2","ip" and "cosine"
  "dim": 23, // dim must integer in [1, 65536]
  "index_param": { // must give this key: "index_param"
    "base_quantization_type": "sq8", /* must, support "sq8", "fp32", "sq8_uniform", "sq4_uniform",
                                        "pq", "pqfs", "fp16", "bf16", "rabitq";
                                        means the quantization type for origin vector data*/
                                        
    "use_reorder": false, /* optional, default false, if set true means use high precise code to reorder,
                             the 'precise_quantization_type' must be set */
                             
    "precise_quantization_type": "fp32", /* optional, when 'use_reorder' is true, this key must be set, 
                                            the value like 'base_quantization_type' by more precise */
                                           
    "max_degree": 32, /* optional, default is 64, means the max_degree for hgraph bottom graph */
    
    "ef_construction": 200,  /* optional, default is 400, means the ef_construct value for hgraph graph */
    
    "hgraph_init_capacity": 100, /* optional, default is 100, means the initial capacity 
                                    when hgraph is created, not real size */
                                    
    "build_thread_count": 100, /* optional, default is 100, means how much thread will be used for hgraph build */
    
    "base_io_type": "block_memory_io", /* optional, default is 'block_memory_io', 
                                          support "memory_io", "block_memory_io", "buffer_io", "async_io", "mmap_io"
                                          means the io type for 'base_quantization' codes 
                                          the "memory_io" and "block_memory_io" mean stored in memory,
                                          "buffer_io" means on local SSD or local disk. "reader_io" means that the 
                                          IO section will be set during deserialization, and IO operations will be 
                                          based on Reader*/

    "base_file_path": "./default_file_path", /* optional, default is "./default_file_path", 
                                                means the filepath for 'base_quantization' codes storage,
                                                the parameter is meaningless 
                                                when base_io_type is "memory_io" or "block_memory_io" */
                                                
    "precise_io_type": "block_memory_io", /* optional, default is 'block_memory_io', 
                                             same as "base_io_type", but for precise codes */
                                             
    "precise_file_path": "./default_file_path", /* optional, default is './default_file_path', 
                                                  same as "base_file_path", but for precise codes */

    "ignore_reorder": false, /* optional, default is false,
                               if set true, means the precise_quantization will be ignored on serialization */

    "build_by_base": false, /* optional, default is false,
                              if set true, means the index will be built by base_quantization codes,
                              if set false, means the index will be built by precise_quantization codes */
                              
    "base_pq_dim": 128, /* optional, when base_quantization_type is "pq" or "pqfs", this key must be set.
                            means the pq subspace count */

    "graph_type": "nsw", /* optional, default is "nsw", support "nsw", "odescent",
                          means the graph type for hgraph */
    "support_duplicate": false, /* optional, default is false, when set to true it adds duplicate data 
                                 checks to reduce the impact of duplicate data on the graph index */
    "store_raw_vector": false, /* optional, default is false, when metric is cosine, set to true to 
                                retrieve raw vectors from the index */
    "use_elp_optimizer": false, /* optional, default is false, automatically adjusts internal parameters 
                                 after index construction or deserialization based on system conditions */
    "support_remove": false  /* optional, default is false, set to true when the index needs to support 
                              deletions */
  }
}
```

### Search Params
The example of the `search_params`.
```json5
{
  "hgraph": {
    "ef_search": 200, /* must, means the ef_search value for hgraph graph */
  }
}
```