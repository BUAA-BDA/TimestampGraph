#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstring>

namespace timestampgraph {

typedef int timestamp_t;

enum Metric {
    L2 = 0,
    COSINE = 1
};

class DataSet {
    
public:

    int dim_;
    size_t n_elements_;
    float* data_;
    Metric metric_;

    DataSet(int dim, int max_elements, char* path, Metric metric = Metric::L2) : dim_(dim) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            std::cout << "open dataset file error: " << path << std::endl;
            exit(-1);
        }

        int dim_from_file;
        in.read((char *)&dim_from_file, 4);
        if (dim_from_file != dim) {
            std::cout << "dim not match" << std::endl;
            exit(-1);
        }

        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t)ss;
        int n = (unsigned)(fsize / (dim + 1) / 4);

        n_elements_ = std::min(n, max_elements);
        data_ = new float[(size_t)n_elements_ * (size_t)dim];
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < n_elements_; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *)(data_ + i * dim), dim * 4);
        }
        in.close();
    }

    ~DataSet() {
        delete[] data_;
    }
};

class TANNSDataset : public DataSet {

public:
    size_t* operation_sequence_;
    size_t operation_count_;
    TANNSDataset(int dim, int max_elements, char* path, char* seq_path) : DataSet(dim, max_elements, path) {
        std::ifstream in(seq_path);
        if (!in.is_open()) {
            std::cout << "open seq file error: " << seq_path << std::endl;
            exit(-1);
        }

        std::vector<size_t> seq;
        while (!in.eof()) {
            size_t idx;
            in >> idx;
            seq.push_back(idx);
        }
        in.close();

        operation_sequence_ = new size_t[seq.size()];
        operation_count_ = seq.size();
        for (size_t i = 0; i < seq.size(); i++) {
            operation_sequence_[i] = seq[i];
        }
    }

    ~TANNSDataset() {
        delete[] operation_sequence_;
    }
};

template<typename dist_t, typename label_t>
class TANNSAlgorithmInterface {
public:   
    virtual const char* name() = 0;

    virtual void addPoint(const void *datapoint, label_t label) = 0;

    virtual void removePoint(label_t label) = 0;

    virtual std::priority_queue<std::pair<dist_t, label_t>> 
        searchKnn(const void *query_data, size_t k, size_t timestamp, size_t ef = 10) = 0; 

    virtual void buildIndex(const TANNSDataset& dataset) {
        throw std::runtime_error("Only For Indexs That Do Not Support Dynamic Data");
    }

    virtual bool supportAddPoint() {
        return true;
    }

    virtual size_t memory_cost() const {
        return -1;
    }
};

void normalize_l2(float* data, size_t dim) {
    float norm = 0;
    for (size_t i = 0; i < dim; i++) {
        norm += data[i] * data[i];
    }
    norm = 1.0 / sqrt(norm);
    for (size_t i = 0; i < dim; i++) {
        data[i] *= norm;
    }
}

const float* copy_and_normalize_l2(const float* data, size_t dim) {
    float* data_copy = new float[dim];
    memcpy(data_copy, data, dim * sizeof(float));
    normalize_l2(data_copy, dim);
    return data_copy;
}

} // namespace timestampgraph