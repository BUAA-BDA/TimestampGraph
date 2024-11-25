#include "timestamp_graph.h"
#include "benchmark_util.h"

using namespace timestampgraph;


int main(int argc, char** argv) {
    TANNSBenchConfig config(argc, argv);
    config.print_header("Single Index Benchmark");
    std::vector<std::unordered_set<size_t>> gts = read_groundtruth(config.get_arg(0));

    size_t M = 16;
    std::unique_ptr<TANNSAlgorithmInterface<float, hnswlib::labeltype>> ptr;
    ptr.reset(new CompressedTimestampGraph<float>(config.train_set, M, 200, M / 2, config.metric));

    std::vector<size_t> efs = { 10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000 };
    std::pair<std::unique_ptr<TANNSAlgorithmInterface<float, hnswlib::labeltype>>, std::vector<size_t>>
        pair(std::move(ptr), std::move(efs));

    TANNSBenchmark<float, hnswlib::labeltype> bench(std::move(pair));
    if (config.get_arg(0) == nullptr) {
        throw std::runtime_error("No GT Path");
    }

    bench.run_build(config);
    bench.run_search(config, gts);
    bench.print_memory_usage();

    return 0;
}