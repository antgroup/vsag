

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
    "base_quantization_type": "sq8", /* must, support "sq8", "fp32", "sq8_uniform", "sq4_uniform";
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
                                          support "memory_io", "block_memory_io", "buffer_io"
                                          means the io type for 'base_quantization' codes 
                                          the "memory_io" and "block_memory_io" mean store in memory,
                                          "buffer_io" means on local SSD or local disk. */
                                        
    "base_file_path": "./default_file_path", /* optional, default is "./default_file_path", 
                                                means the filepath for 'base_quantization' codes storage,
                                                the parameter is meaningless 
                                                when base_io_type is "memory_io" or "block_memory_io" */
                                                
    "precise_io_type": "block_memory_io", /* optional, default is 'block_memory_io', 
                                             same as "base_io_type", but for precise codes */
                                             
    "precise_file_path": "./default_file_path" /* optional, default is './default_file_path', 
                                                  same as "base_file_path", but for precise codes */
  }
}
```