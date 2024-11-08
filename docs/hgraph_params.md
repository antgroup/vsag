

The document introduces how to modify the `factory json` for HGraph Index.

Different capabilities of HGraph Index can be enabled by configuring various parameter combinations.

You can use `vsag::Factory::CreateIndex("hgraph", hgraph_json_string)` to factory HGraph Index

The following will introduce how to edit the `hgraph_json_string` like this

### Template

The template of the `hgraph_json_string`.
```json5
{
  "dtype": "float32", // data_type: only support float32 for hgraph
  "metric_type": "l2", // metric_type only support "l2","ip" and "cosine"
  "dim": 23, // dim must integer in [1, 65536]
  "index_param": { // must give this key: "index_param"
    "key": value,
    ...
  }
}
```
### Index parameters

We will introduce some terms

- `[key]` means the string is a key.
- `[value]` means the string is a value.
- `[must]` means the key is must given.

The following strings are the detail of the index_param config

#### `"base_quantization_type"` [key] [must]
- means the quantization of the base_code
- the value of this key is a $string$
- current support `"sq8"` and `"fp32"`
- related keys: `"sq8"`,`"fp32"`

#### `"use_reorder"` [key]
- means enable use high precise codec to reorder the result
- the value of this key is a $bool$
- the default value is **true**

#### `"sq8"` [value]
- means use sq8 quantization
- is the value of the `"base_quantization_type"`
- related keys: `"base_quantization_type"`

#### `"fp32"` [value]
- means use fp32 quantization (actually the origin data_type)
- is the value of the `"base_quantization_type"`
- related keys: `"base_quantization_type"`


