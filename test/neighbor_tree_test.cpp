#include "neighbor_tree.h"
#include <random>
#include <set>

using timestampgraph::tableint;
using timestampgraph::NeighborTree;

auto gen_versions(size_t versions, size_t elements, size_t delta, bool overlap, size_t overlap_chance, bool verbose, size_t seed = 2024) -> std::vector<std::set<tableint>> {
    std::vector<std::set<tableint>> ret;
    std::set<tableint> cur_in;
    std::set<tableint> cur_out;

    for (size_t i = 0; i < elements; i++) {
        cur_out.insert(i);
    }

    std::srand(seed);

    for (size_t i = 0; i < versions; i++) {
        std::set<tableint> delta_in;
        std::set<tableint> delta_out;
        if (verbose) std::cout << "Generating version " << i << ": ";
        // curin_0 = curin_1 + deltaout_1
        // curout_0 = curout_1 + deltain_1
        for (size_t j = 0; j < delta; j++) {
            bool can_add = !cur_out.empty() && (cur_out.size() - delta_in.size() > 0);
            bool can_remove = cur_in.size() - delta_out.size() > 0;
            bool exit = !can_add && !can_remove;
            bool add = can_add && (!can_remove || rand() % 2);
            if (exit) {
                if (verbose) std::cout << "BLANK";
                break;
            }
            else if (add) {
                size_t idx;
                tableint elem;
                do {
                    idx = rand() % cur_out.size();
                    elem = *std::next(cur_out.begin(), idx);
                } while (delta_in.find(elem) != delta_in.end());
                delta_in.insert(elem);
                if (verbose) std::cout << "+" << elem << " ";
            }
            else {
                size_t idx;
                tableint elem;
                do {
                    idx = rand() % cur_in.size();
                    elem = *std::next(cur_in.begin(), idx);
                } while (delta_out.find(elem) != delta_out.end());
                delta_out.insert(elem);
                if (verbose) std::cout << "-" << elem << " ";
            }
        }

        if (verbose) std::cout << std::endl;
        for (tableint elem : delta_in) {
            cur_in.insert(elem);
            cur_out.erase(elem);
        }
        for (tableint elem : delta_out) {
            cur_in.erase(elem);
            if (overlap && rand() % 100 < overlap_chance) {
                cur_out.insert(elem);
            }
        }

        std::set<tableint> cur_version = std::set<tableint>(cur_in);
        ret.push_back(cur_version);
        if (verbose) {
            std::cout << "Generated Version " << i << ":";
            for (tableint elem : cur_version) {
                std::cout << " " << elem;
            }
            std::cout << "(";
            for (tableint elem : cur_out) {
                std::cout << " " << elem;
            }
            std::cout << ")" << std::endl;
        }

        if (cur_in.empty() && cur_out.empty()) {
            break;
        }
    }

    return ret;
}

void validation(std::vector<std::set<tableint>>& data, size_t node_size, bool verbose) {
    NeighborTree vt(node_size);

    for (size_t i = 0; i < data.size(); i++) {
        if (verbose) {
            std::cout << "Version " << i << ":";
            for (tableint elem : data[i]) {
                std::cout << " " << elem;
            }
            std::cout << std::endl;
        }
        std::set<tableint> version = data[i];
        std::vector<tableint> top_candidates;
        for (tableint elem : version) {
            top_candidates.push_back(elem);
        }
        vt.set_linklist(i, top_candidates);

        for (size_t j = 0; j <= i; j++) {
            std::vector<tableint> result = vt.get_linklist(j);
            std::set<tableint> expected(data[j]);
            std::set<tableint> actual(result.begin(), result.end());
            if (expected != actual) {
                std::cerr << "Error: version " << j << " mismatch on iteration " << i << std::endl;
                auto ll = vt.get_linklist(j);
                for (tableint elem : expected) {
                    if (actual.find(elem) == actual.end()) {
                        std::cerr << "Missing element " << elem << std::endl;
                    }
                }
                for (tableint elem : actual) {
                    if (expected.find(elem) == expected.end()) {
                        std::cerr << "Unexpected element " << elem << std::endl;
                    }
                }
                throw std::runtime_error("Validation failed");
            }
        }
        if (verbose) {
            std::cout << "Version " << i << " passed" << std::endl;
        }
    }
}

void performance(std::vector<std::set<tableint>>& data, size_t node_size, size_t max_repeat = 100) {
    size_t build_cost = 0;
    size_t query_cost = 0;

    for (size_t repeat = 0; repeat < max_repeat; repeat++) {
        NeighborTree vt(node_size);

        for (size_t i = 0; i < data.size(); i++) {
            std::set<tableint> version = data[i];
            std::vector<tableint> top_candidates;
            for (tableint elem : version) {
                top_candidates.push_back(elem);
            }
            auto start = std::chrono::high_resolution_clock::now();
            vt.set_linklist(i, top_candidates);
            build_cost += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        }

        for (size_t i = 0; i < data.size(); i++) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<tableint> result = vt.get_linklist(i);
            query_cost += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

            std::set<tableint> expected(data[i]);
            std::set<tableint> actual(result.begin(), result.end());
            if (expected != actual) {
                std::cerr << "Error: version " << i << " mismatch" << std::endl;
                auto ll = vt.get_linklist(i);
                for (tableint elem : expected) {
                    if (actual.find(elem) == actual.end()) {
                        std::cerr << "Missing element " << elem << std::endl;
                    }
                }
                for (tableint elem : actual) {
                    if (expected.find(elem) == expected.end()) {
                        std::cerr << "Unexpected element " << elem << std::endl;
                    }
                }
                throw std::runtime_error("Validation failed");
            }
        }
    }


    std::cout << "Node Size: " << node_size << ", Build Cost: " << build_cost / max_repeat << "us, Query Cost: " << query_cost / max_repeat << "us" << std::endl;
}

int main() {

    size_t versions = 1000;
    size_t elements = 800;
    size_t delta = 4;
    bool overlap = false;
    size_t overlap_chance = 30;

    size_t node_size = 3;

    bool verbose = false;

    std::cout << "Begin Test with param: {versions = " << versions << ", elements = " << elements << ", delta = " << delta << ", overlap = " << overlap << ", overlap_chance = " << overlap_chance << ", node_size = " << node_size << "}" << std::endl;
    auto data = gen_versions(versions, elements, delta, overlap, overlap_chance, verbose);
    std::cout << "Generated data, Start Validation" << std::endl;
    validation(data, node_size, verbose);
    std::cout << "Validation Passed, Start Performance Test" << std::endl;
    for (size_t i = 1; i < 100; i++) {
        performance(data, i);
    }
}