#include "interface.h"
#include <chrono>
#include <iomanip>
#include <unordered_set>

namespace timestampgraph {

class TANNSBenchConfig {
public:
    int argc_;
    char** argv_;
    int dim_;
    int n_elements;
    int max_elements;
    int n_queries;
    int max_queries;
    char* data_path;
    char* query_path;
    char* op_seq_path;
    char* query_seq_path;
    int k;
    Metric metric;
    TANNSDataset train_set;
    TANNSDataset query_set;

    TANNSBenchConfig(int argc, char **argv) : argc_(argc), argv_(argv),
        dim_(std::stoi(argv[1])), k(std::stoi(argv[3])),
        max_elements(std::stoi(argv[2])), data_path(argv[4]), op_seq_path(argv[7]), 
        train_set(dim_, max_elements, data_path, op_seq_path), 
        max_queries(std::stoi(argv[6])), query_path(argv[5]), query_seq_path(argv[8]),
        query_set(dim_, max_queries, query_path, query_seq_path) {

        if (argc < 9) {
            std::cout << "Usage: " << argv[0] << " ";
            std::cout << "dim max_elements k data_path query_path query_num op_seq_path query_seq_path" << std::endl;
            std::cout << "Current: " << argc << std::endl;
            exit(1);
        }

        n_elements = train_set.n_elements_;
        n_queries = query_set.n_elements_;

        std::string train_set_path(argv[4]);
        metric = (train_set_path.find("glove") == std::string::npos) ? Metric::L2 : Metric::COSINE;
    }

    void print_header(const char* description) const {
        time_t current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        std::cout << "======================================" << std::endl;
        std::cout << description << " At " << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Dim: " << dim_ << ", Elements: " << n_elements << "/" << max_elements << ", Query: " << n_queries << "/" << max_queries << ", K: " << k << std::endl;
        std::cout << "Operation Sequence:\t" << " (" << train_set.operation_count_ << ")\"" << op_seq_path << "\"" << std::endl;
        std::cout << "Query Sequence:    \t" << " (" << query_set.operation_count_ << ")\"" << query_seq_path << "\"" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Data Path: " << data_path << std::endl;
        std::cout << "Query Path: " << query_path << std::endl;
        std::cout << "Metric: " << (metric == Metric::L2 ? "L2" : "COSINE") << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Original Command: ";
        for (int i = 0; i < argc_; i++) {
            std::cout << argv_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

    char* get_arg(int idx) {
        return (idx < argc_ - 9) ? argv_[idx + 9] : nullptr; 
    }
};

template<typename dist_t, typename label_t>
class TANNSBenchmark {
    std::unique_ptr<TANNSAlgorithmInterface<dist_t, label_t>> algorithm_;

    struct BuildRecord {
        size_t insert_time_, insert_count_, last_verbose_insert_count_;
        size_t remove_time_, remove_count_, last_verbose_remove_count_;
        size_t build_time_;

        inline void record_insert(size_t time) {
            insert_time_ += time;
            build_time_ += time;
            insert_count_++;
        }

        inline void record_remove(size_t time) {
            remove_time_ += time;
            build_time_ += time;
            remove_count_++;
        }

        inline void verbose(size_t count) {
            bool report_insert = (insert_count_ % count == 0) && (insert_count_ != last_verbose_insert_count_);
            bool report_remove = (remove_count_ % count == 0) && (remove_count_ != last_verbose_remove_count_);
            if (report_insert || report_remove) {
                std::cout << "\tProgress: Inserted " << insert_count_ << " in " << insert_time_ << " us; Removed " << remove_count_ << " in " << remove_time_ << " us; Total: " << build_time_ << " us" << std::endl;
                last_verbose_insert_count_ = insert_count_;
                last_verbose_remove_count_ = remove_count_;
            }
        }
    } build_record_;

    struct QueryRecord {
        size_t query_time_;
        size_t query_hits_;
        size_t query_total_elements_;
    };

    std::vector<QueryRecord> query_records_;
    std::vector<size_t> query_efs_;

public:
    TANNSBenchmark(std::pair<std::unique_ptr<TANNSAlgorithmInterface<dist_t, label_t>>, std::vector<size_t>>&& pair) 
        : algorithm_(std::move(pair.first)), query_efs_(std::move(pair.second)) {
        query_records_.resize(query_efs_.size(), {0, 0, 0});
        memset(&build_record_, 0, sizeof(build_record_));
    }

    void run_build(const TANNSBenchConfig& config) {
        std::cout << "Building Index(" << algorithm_->name() << ")..." << std::endl;
        auto& train_set = config.train_set;
        auto& query_set = config.query_set;

        std::unordered_set<size_t> points;
        if (!algorithm_->supportAddPoint()) {
            auto start = std::chrono::high_resolution_clock::now();
            algorithm_->buildIndex(config.train_set);
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            build_record_.build_time_ = time;
        } else {
            for (size_t j = 0; j < train_set.operation_count_; j++) {
                size_t idx = train_set.operation_sequence_[j];
                if (idx >= train_set.n_elements_) {
                    continue;
                }
                
                bool is_insert = points.find(idx) == points.end();
                auto start = std::chrono::high_resolution_clock::now();
                if (is_insert) {
                    algorithm_->addPoint(train_set.data_ + idx * train_set.dim_, idx);
                    points.insert(idx);
                } else {
                    algorithm_->removePoint(idx);
                }
                auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
                if (is_insert) {
                    build_record_.record_insert(time);
                } else {
                    build_record_.record_remove(time);
                }
                build_record_.verbose(train_set.n_elements_ / 5000);
            }
        }
            
        std::cout << "Build Time(" << algorithm_->name() << "): " << build_record_.build_time_ << \
            " us; Insert Time: " << build_record_.insert_time_ << \
            " us; Remove Time: " << build_record_.remove_time_ << " us" << std::endl;    
        std::cout << "--------------------------------------" << std::endl;
    }


    void run_search(const TANNSBenchConfig& config, const std::vector<std::unordered_set<size_t>>& ground_truths) {


        if (config.n_queries != ground_truths.size()) {
            std::cout << "Ground Truths Size(" << ground_truths.size() << ") != Query Size(" << config.n_queries << ")" << std::endl;
            exit(1);
        }
        
        for (int i = 0; i < config.n_queries; i++) {
            if (i % (config.n_queries / 10) == 0 && i != config.n_queries && i != 0) {
                std::cout << "Progress: " << i << "/" << config.n_queries << std::endl;
                for (int j = 0; j < query_efs_.size(); j++) {
                    auto& record = query_records_[j];
                    auto ef = query_efs_[j];
                    std::cout << "Search Time Current Avg(" << algorithm_->name() << ", ef=" << ef << "): "
                        << record.query_time_ / i << " us" << std::endl;
                    std::cout << "Current Recall(" << algorithm_->name() << ", ef=" << ef << "): " 
                        << record.query_hits_ << "/" << record.query_total_elements_ << " = " 
                        << record.query_hits_ * 1.0 / record.query_total_elements_ << std::endl;
                }
                std::cout << "--------------------------------------" << std::endl;
            }


            float* query = config.query_set.data_ + i * config.query_set.dim_;
            auto ts = config.query_set.operation_sequence_[i];
            auto& ground_truth = ground_truths[i];
            
            for (int j = 0; j < query_efs_.size(); j++) {
                auto ef = query_efs_[j];
                auto& record = query_records_[j];
                
                auto start = std::chrono::high_resolution_clock::now();
                std::priority_queue<std::pair<dist_t, label_t>> result = algorithm_->searchKnn(query, config.k, ts, ef);
                auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

                record.query_time_ += time;
                record.query_total_elements_ += config.k;

                while (result.size() > 0) {
                    size_t idx = result.top().second;
                    result.pop();
                    if (ground_truth.find(idx) != ground_truth.end()) {
                        record.query_hits_++;
                    }
                }
            }
        }
        
        for (int i = 0; i < query_efs_.size(); i++) {
            auto& record = query_records_[i];
            auto ef = query_efs_[i];
            std::cout << "Search Time Avg(" << algorithm_->name() << ", ef=" << ef << "): "
                << record.query_time_ / config.n_queries << " us" << std::endl;
            std::cout << "Recall(" << algorithm_->name() << ", ef=" << ef << "): " 
                << record.query_hits_ << "/" << record.query_total_elements_ << " = " 
                << record.query_hits_ * 1.0 / record.query_total_elements_ << std::endl;
        }
        std::cout << "--------------------------------------" << std::endl;
    }

    void run_generate_answer(const TANNSBenchConfig& config, const char* path) {


        static_assert(sizeof(int) == 4, "Int Size Error");      


        std::ofstream file(path, std::ios::binary);
        for (int i = 0; i < config.n_queries; i++) {
            float* query = config.query_set.data_ + i * config.query_set.dim_;
            auto ts = config.query_set.operation_sequence_[i];


            std::vector<int> answer;
            {
                int j = 0;
                auto ef = query_efs_[j];
                auto& record = query_records_[j];
                
                auto start = std::chrono::high_resolution_clock::now();
                std::priority_queue<std::pair<dist_t, label_t>> result = algorithm_->searchKnn(query, config.k, ts, ef);
                auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

                record.query_time_ += time;
                record.query_total_elements_ += config.k;

                while (result.size() > 0) {
                    answer.push_back(result.top().second);
                    result.pop();
                }
            }

            if (answer.size() != config.k) {
                std::cout << "Answer Size(" << answer.size() << ") != K(" << config.k << ")" << std::endl;
                exit(1);
            }
            file.write((char*)&config.k, 4);
            while (!answer.empty()) {
                int idx = answer.back();
                answer.pop_back();
                file.write((char*)&idx, 4);
            }
        }

        {   
            int i = 0;
            auto& record = query_records_[i];
            auto ef = query_efs_[i];
            std::cout << "Search Time Avg(" << algorithm_->name() << ", ef=" << ef << "): "
                << record.query_time_ / config.n_queries << " us" << std::endl;
            std::cout << "Recall(" << algorithm_->name() << ", ef=" << ef << "): " 
                << record.query_hits_ << "/" << record.query_total_elements_ << " = " 
                << record.query_hits_ * 1.0 / record.query_total_elements_ << std::endl;
        }
        std::cout << "--------------------------------------" << std::endl;
    }

    void print_memory_usage() {
        std::ifstream file_stream("/proc/self/status");
        std::string line;
        while (std::getline(file_stream, line)) {
            if (line.find("VmHWM") != std::string::npos) {
                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                size_t value = std::stoull(line.substr(begin, end - begin + 1));
                std::cout << "Peak PMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
            }
            if (line.find("VmRSS") != std::string::npos) {
                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                size_t value = std::stoull(line.substr(begin, end - begin + 1));
                std::cout << "Current PMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
            }
            if (line.find("VmPeak") != std::string::npos) {

                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                size_t value = std::stoull(line.substr(begin, end - begin + 1));
                std::cout << "Peak VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
            }
            if (line.find("VmSize") != std::string::npos) {

                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                size_t value = std::stoull(line.substr(begin, end - begin + 1));
                std::cout << "Current VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
            }
            if (line.find("VmData") != std::string::npos) {

                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                size_t value = std::stoull(line.substr(begin, end - begin + 1));
                std::cout << "Data Segment VMemory Usage:" << std::to_string(value / 1024.0) + " MB" << std::endl;
            }
        }
        std::cout << "--------------------------------" << std::endl;
        algorithm_->memory_cost();
        std::cout << "================================" << std::endl;
    }
};


auto read_groundtruth(char* path) -> std::vector<std::unordered_set<size_t>> {

    std::ifstream file_stream(path, std::ios::binary);
    if (!file_stream.is_open()) {
        std::cout << "Open GroundTruth File Error: " << path << std::endl;
        exit(1);
    }
    
    std::vector<std::unordered_set<size_t>> ground_truths;
    while (!file_stream.eof()) {
        int num;
        file_stream.read((char*)&num, 4);
        if (file_stream.fail() && file_stream.eof()) {
            break;
        }
        assert(num == 10);
        std::unordered_set<size_t> ground_truth;
        for (int i = 0; i < num; i++) {
            int idx;
            file_stream.read((char*)&idx, 4);
            ground_truth.emplace(idx);
        }
        ground_truths.push_back(ground_truth);
    }
    return ground_truths;
}

} // namespace timestampgraph