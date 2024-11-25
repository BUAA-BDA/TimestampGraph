## Timestamp Graph

Timestamp Graph is an index for Timestamp Approximate Nearest Neighbor Search.

The implementation of the Timestamp Graph is built on [hnswlib](https://github.com/nmslib/hnswlib).

### Usage Example

```cpp
#include "timestamp_graph.h"

// Load the dataset
int dim = 96;
int max_elements = 100000;
const char* data_path = "path_to_deep_base.fvecs";
const char* op_seq_path = "path_to_operation_sequence.txt";

TANNSDataset train_set(dim, max_elements, data_path, op_seq_path);

// Create the algorithm
int M = 16;
int ef_construction = 200;
int node_size = 8;

auto algorithm = new CompressedTimestampGraph<float>(train_set, M, ef_construction, node_size, Metric::L2);

```

### Run Example Tests

To build the example, you can clone this repository and run the following commands:

```bash
cd timestamp-graph
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Running examples on the DEEP dataset:

```bash
./timestamp_graph_test 96 100000 10 [path_to_deep_base.fvecs] [path_to_deep_query.fvecs] 10000 [path_to_operation_timestamps.txt] [path_to_query_timestamps.txt] [path_to_TANNS_groundtruth.ivecs]

./timestamp_graph_test [dimension] [base_set_size] [search_k] [path_to_deep_base.fvecs] [path_to_deep_query.fvecs] [query_set_size] [path_to_operation_timestamps.txt] [path_to_query_timestamps.txt] [path_to_TANNS_groundtruth.ivecs]
```
