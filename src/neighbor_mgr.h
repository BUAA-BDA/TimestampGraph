#pragma once
#include <hnswlib/hnswlib.h>

namespace timestampgraph {

typedef unsigned int tableint;
typedef int timestamp_t;
typedef std::pair<timestamp_t, timestamp_t> interval_t;
using hnswlib::linklistsizeint;

class SpanMgr {

private:
    std::unordered_map<tableint, interval_t> element_span_;

protected:
    size_t max_elements_{ 0 };

    SpanMgr(size_t max_elements) : max_elements_(max_elements) {
        element_span_.reserve(max_elements);
    }

    inline void set_birth(tableint internal_id, timestamp_t ts) {
        element_span_[internal_id] = std::make_pair(ts, std::numeric_limits<timestamp_t>::max());
    }

public:
    inline void set_death(tableint internal_id, timestamp_t ts) {
        element_span_[internal_id].second = ts;
    }

    template<bool tolerate_ep = false>
    inline bool check_alive(tableint internal_id, timestamp_t ts) const {
        if (tolerate_ep && internal_id == 0)
            return true;
        timestamp_t birth = element_span_.at(internal_id).first;
        timestamp_t death = element_span_.at(internal_id).second;
        return birth <= ts && ts < death;
    }

    inline size_t memory_cost() const {
        return sizeof(SpanMgr) + max_elements_ * sizeof(std::pair<tableint, interval_t>);
    }
};


template<typename nodeid_t = tableint>
class NeighborsMgr : public SpanMgr {

private:
    typedef std::vector<nodeid_t> neighbors_t;
    typedef std::pair<nodeid_t, neighbors_t> neighbors_pair_t;
    std::vector<std::vector<neighbors_pair_t>>* neighbors_history_;
    size_t node_size_, node_size0_, max_elements_;

    std::vector<linklistsizeint> element_levels_;

public:

    NeighborsMgr(size_t max_elements, size_t node_size, size_t node_size0)
        : SpanMgr(max_elements), node_size_(node_size), node_size0_(node_size0), max_elements_(max_elements) {
        std::allocator<std::vector<std::vector<neighbors_pair_t>>> alloc;
        neighbors_history_ = alloc.allocate(max_elements_);
        for (size_t i = 0; i < max_elements_; i++) {
            alloc.construct(neighbors_history_ + i, std::vector<std::vector<neighbors_pair_t>>(5));
        }
        element_levels_.resize(max_elements_, 0);
    }

    ~NeighborsMgr() {
        std::allocator<std::vector<std::vector<neighbors_pair_t>>> alloc;
        for (size_t i = 0; i < max_elements_; i++) {
            alloc.destroy(neighbors_history_ + i);
        }
        alloc.deallocate(neighbors_history_, max_elements_);
    }

    inline const std::vector<nodeid_t> get_linklist0(nodeid_t internal_id, timestamp_t ts) const {
        return get_linklist(internal_id, 0, ts);
    }

    inline const std::vector<nodeid_t> get_linklist1(nodeid_t internal_id, int level, timestamp_t ts) const {
        return get_linklist(internal_id, level, ts);
    }


    inline const std::vector<nodeid_t> get_linklist(nodeid_t internal_id, int level, timestamp_t ts) const {
        assert(0 <= internal_id && internal_id < max_elements_);
        const std::vector<neighbors_pair_t>& neighbors = neighbors_history_[internal_id].at(level);
        auto it = std::upper_bound(neighbors.begin(), neighbors.end(), ts,
            [](timestamp_t ts, const neighbors_pair_t& p) { return ts < p.first; });
        if (it == neighbors.begin()) {
            return {};
        }
        else {
            return (it - 1)->second;
        }
    }

    inline void alloc_linklist(nodeid_t internal_id, timestamp_t ts, int level) {
        assert(0 <= internal_id && internal_id < max_elements_);
        auto& neighbors = neighbors_history_[internal_id];
        if (neighbors.size() != 5) {
            throw std::runtime_error("neighbors_history_[" + std::to_string(internal_id) + "] is not empty");
        }
        neighbors.resize(level + 1);
        element_levels_[internal_id] = level;
        set_birth(internal_id, ts);
    }

    inline void check_valid(nodeid_t internal_id, int level, timestamp_t ts, linklistsizeint expect_size = -1) const {
        auto ll = get_linklist(internal_id, level, ts);
        if (expect_size >= 0 && ll.size() != expect_size)
            throw std::runtime_error("check_valid: size error");
        for (nodeid_t i : ll) {
            if (i < 0 || i > max_elements_)
                throw std::runtime_error("check_valid: neighbor error");
        }
    }

    inline linklistsizeint element_level(nodeid_t internal_id) const {
        return element_levels_[internal_id];
    }

    inline void append_linklist(nodeid_t internal_id, int level, nodeid_t val, timestamp_t ts) {
        assert(0 <= internal_id && internal_id < max_elements_);
        assert(0 <= level && level < neighbors_history_[internal_id].size());
        auto& last_neighbors = get_linklist(internal_id, level, ts);
        auto new_neighbors = last_neighbors;
        new_neighbors.push_back(val);
        neighbors_history_[internal_id][level].push_back(std::make_pair(ts, new_neighbors));
    }

    inline void set_linklist(nodeid_t internal_id, int level, const std::vector<nodeid_t>& list, timestamp_t ts) {
        auto new_neighbors = list;
        neighbors_history_[internal_id][level].push_back(std::make_pair(ts, new_neighbors));
    }

    inline size_t memory_cost() const {
        size_t cost = SpanMgr::memory_cost() + sizeof(NeighborsMgr) - sizeof(SpanMgr);

        cost += max_elements_ * sizeof(std::vector<std::vector<neighbors_pair_t>>*);
        for (size_t i = 0; i < max_elements_; i++) {
            cost += neighbors_history_[i].capacity() * sizeof(std::vector<neighbors_pair_t>);
            std::vector<std::vector<neighbors_pair_t>>& neighbors = neighbors_history_[i];
            for (const auto& level_neighbors : neighbors) {
                cost += level_neighbors.capacity() * sizeof(neighbors_pair_t);
                for (const auto& pair : level_neighbors) {
                    cost += pair.second.capacity() * sizeof(nodeid_t);
                }
            }
        }

        return cost;
    }
};

template<typename nodeid_t>
class PossibleNeighborsMgr {

private:

    std::vector<std::vector<nodeid_t>> possible_neighbors_;
    std::vector<int> possible_neighbors_size_;
    size_t node_size_;

public:

    PossibleNeighborsMgr(size_t max_elements, size_t node_size) : node_size_(node_size) {
        possible_neighbors_.resize(max_elements);
        possible_neighbors_size_.resize(max_elements, 0);
        for (size_t i = 0; i < max_elements; i++) {
            possible_neighbors_[i].resize(node_size, -1);
        }
    }

    inline void set_possible_neighbor(nodeid_t internal_id, nodeid_t val) {
        int& size = possible_neighbors_size_[internal_id];
        possible_neighbors_[internal_id][(size++) % node_size_] = val;
    }

    inline void set_possible_neighbors(nodeid_t internal_id, std::vector<nodeid_t> vals) {
        int& size = possible_neighbors_size_[internal_id];
        if (vals.size() >= node_size_) {
            size += node_size_;
            std::copy(vals.begin(), vals.begin() + node_size_, possible_neighbors_[internal_id].begin());
        }
        else for (nodeid_t val : vals) {
            possible_neighbors_[internal_id][(size++) % node_size_] = val;
        }
    }

    inline const std::vector<nodeid_t> get_possible_neighbors(nodeid_t internal_id, timestamp_t ts) {
        if (possible_neighbors_size_[internal_id] < node_size_) {
            return std::vector<nodeid_t>(possible_neighbors_[internal_id].begin(), possible_neighbors_[internal_id].begin() + possible_neighbors_size_[internal_id]);
        }
        return possible_neighbors_[internal_id];
    }

    inline size_t memory_cost() const {
        size_t cost = sizeof(PossibleNeighborsMgr);
        cost += possible_neighbors_size_.capacity() * sizeof(int);
        cost += possible_neighbors_.capacity() * sizeof(std::vector<nodeid_t>);
        for (const auto& neighbors : possible_neighbors_) {
            cost += neighbors.capacity() * sizeof(nodeid_t);
        }
        return cost;
    }
};

template<typename nodeid_t = tableint>
class NeighborsMgrWithPossibleNeighbors : public NeighborsMgr<nodeid_t> {

private:

    PossibleNeighborsMgr<nodeid_t> backup_;

public:

    NeighborsMgrWithPossibleNeighbors(size_t max_elements, size_t node_size, size_t node_size0)
        : NeighborsMgr<nodeid_t>(max_elements, node_size, node_size0), backup_(max_elements, node_size / 2) {}

    inline void set_possible_neighbor(nodeid_t internal_id, nodeid_t val) {
        backup_.set_possible_neighbor(internal_id, val);
    }

    inline void set_possible_neighbors(nodeid_t internal_id, std::vector<nodeid_t> vals) {
        backup_.set_possible_neighbors(internal_id, vals);
    }

    inline const std::vector<nodeid_t> get_possible_neighbors(nodeid_t internal_id, timestamp_t ts) {
        return backup_.get_possible_neighbors(internal_id, ts);
    }

    inline size_t memory_cost() const {
        size_t cost = NeighborsMgr<nodeid_t>::memory_cost();
        std::cout << "Nbr cost: " << cost << std::endl;
        size_t bkcost = backup_.memory_cost();
        std::cout << "PossibleNbr cost: " << bkcost << std::endl;
        return cost + bkcost;
    }
};

} // namespace timestampgraph
