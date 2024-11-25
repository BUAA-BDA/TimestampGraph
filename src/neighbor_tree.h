#pragma once
#include <hnswlib/hnswlib.h>
#include <stack>
#include "neighbor_tree_utils.h"

namespace timestampgraph {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
typedef int timestamp_t;

class NeighborTree {

    struct DeadElement {
        tableint idx;
        timestamp_t born;
        timestamp_t dead;
    };

    class Node {
    public:
        timestamp_t split_;
        nodeidx_t level_;
        nodeidx_t parent_;
        std::vector<DeadElement> nodes_;

    public:

        Node(timestamp_t split, nodeidx_t idx, size_t expect_size) : split_(split) {
            level_ = calc_level(idx);
            parent_ = calc_parent(idx);
        }

        void search(std::vector<tableint>& dest, timestamp_t timestamp) const {
            for (auto node : nodes_) {
                if (node.born <= timestamp && timestamp < node.dead) {
                    dest.push_back(node.idx);
                }
            }
        }

        bool insert(tableint idx, interval_t interval) {
            if (split_ < 0 || (interval.first <= split_ && split_ <= interval.second)) {
                nodes_.push_back({ idx, interval.first, interval.second });
                return true;
            }
            return false;
        }

        bool remove(tableint idx) {
            for (size_t i = 0; i < nodes_.size(); i++) {
                if (nodes_[i].idx == idx) {
                    nodes_.erase(nodes_.begin() + i);
                    return true;
                }
            }
            return false;
        }

        inline size_t size() const {
            return nodes_.size();
        }

        inline nodeidx_t level() const {
            return level_;
        }

        inline timestamp_t split() const {
            return split_;
        }

        inline nodeidx_t parent() const {
            return parent_;
        }

        inline size_t memory_cost() const {
            return sizeof(Node) + nodes_.capacity() * sizeof(DeadElement);
        }
    };

private:

    nodeidx_t node_size_;
    nodeidx_t current_root_;
    std::vector<Node> nodes_;
    std::vector<std::pair<tableint, nodeidx_t>> dead_elements_pos_;
    std::unordered_map<tableint, timestamp_t> alive_elements_;

public:

    NeighborTree(size_t node_size) : node_size_(node_size), current_root_(0) {
        nodes_.emplace_back(-1, 0, node_size);
    }

private:

    auto get_path(nodeidx_t idx) const -> std::vector<nodeidx_t> {
        std::stack<nodeidx_t> path;
        nodeidx_t cur_node = idx;

        while (cur_node != current_root_) {
            assert(cur_node < nodes_.size());
            nodeidx_t parent = nodes_[cur_node].parent();
            while (parent >= nodes_.size()) {
                parent = calc_parent(parent);
            }
            path.push(cur_node);
            cur_node = parent;
        }
        path.push(current_root_);

        std::vector<nodeidx_t> ret;
        while (path.size() > 1) {
            ret.push_back(path.top());
            path.pop();
        }
        return ret;
    }

    auto get_lift(nodeidx_t idx, size_t timestamp) const -> nodeidx_t {
        nodeidx_t cur_node = idx;
        while (cur_node != current_root_) {
            assert(cur_node < nodes_.size());
            nodeidx_t parent = nodes_[cur_node].parent();
            while (parent >= nodes_.size()) {
                parent = calc_parent(parent);
            }
            if (nodes_[parent].split() <= timestamp) {
                return cur_node;
            }
            cur_node = parent;
        }
        return current_root_;
    }

    nodeidx_t insert_interval(tableint idx, interval_t interval) {
        std::vector<nodeidx_t> path = get_path(nodes_.size() - 1);

        for (nodeidx_t i : path) {
            assert(i < nodes_.size());
            if (nodes_[i].insert(idx, interval)) {
                return i;
            }
        }

        if (nodes_.back().insert(idx, interval) == false) {
            throw std::runtime_error("Insert failed");
        }

        return nodes_.size() - 1;
    }

    nodeidx_t lift_interval(tableint idx, nodeidx_t pos, interval_t interval) {
        nodeidx_t lift = get_lift(pos, interval.second);
        assert(lift < nodes_.size());
        assert(pos < nodes_.size());
        if (lift != pos) {
            if (nodes_[pos].remove(idx) == false) {
                throw std::runtime_error("Remove failed");
            }
            if (nodes_[lift].insert(idx, interval) == false) {
                throw std::runtime_error("Insert failed");
            }
        }
        return lift;
    }

    void spawn_node(timestamp_t timestamp) {
        size_t size = nodes_.back().size();
        if (size >= node_size_) {
            nodeidx_t high_node_idx = nodes_.size();
            nodes_.push_back(Node(timestamp + 1, high_node_idx, node_size_));
            nodeidx_t low_node_idx = high_node_idx + 1;
            nodes_.push_back(Node(-1, low_node_idx, node_size_));
            if (nodes_[high_node_idx].level() > nodes_[current_root_].level()) {
                current_root_ = high_node_idx;
            }
        }
    }

    void remove_element(tableint idx, timestamp_t born, timestamp_t timestamp) {
        assert(alive_elements_.find(idx) == alive_elements_.end());

        auto elem = std::upper_bound(dead_elements_pos_.begin(), dead_elements_pos_.end(), idx,
            [](tableint idx, const std::pair<tableint, nodeidx_t>& pair) { return idx < pair.first; });

        bool found = (elem != dead_elements_pos_.end()) && (elem->first == idx);

        if (found) {
            nodeidx_t new_pos = lift_interval(idx, elem->second, std::make_pair(born, timestamp));
            elem->second = new_pos;
        }
        else {
            nodeidx_t new_pos = insert_interval(idx, std::make_pair(born, timestamp));
            dead_elements_pos_.insert(elem, { idx, new_pos });
        }
    }

    void add_element(tableint idx, timestamp_t timestamp) {
        alive_elements_.emplace(idx, timestamp);
    }

public:

    void set_linklist(timestamp_t timestamp, tableint add) {
        if (alive_elements_.find(add) != alive_elements_.end()) {
            return;
        }
        add_element(add, timestamp);
    }


    void set_linklist(timestamp_t timestamp, const std::vector<tableint>& top_candidates) {
        std::unordered_map<tableint, timestamp_t> still_alive;
        std::unordered_map<tableint, timestamp_t>& remain_will_dead = alive_elements_;

        for (tableint idx : top_candidates) {
            if (alive_elements_.find(idx) != alive_elements_.end()) {
                still_alive.emplace(idx, alive_elements_[idx]);
                remain_will_dead.erase(idx);
            }
            else {
                still_alive.emplace(idx, timestamp);
            }
        }

        std::unordered_map<tableint, timestamp_t> dead_elements = std::move(remain_will_dead);
        alive_elements_ = std::move(still_alive);

        for (const auto& pair : dead_elements) {
            remove_element(pair.first, pair.second, timestamp);
        }

        spawn_node(timestamp);
    }

    auto get_linklist(timestamp_t timestamp) -> std::vector<tableint> {
        std::vector<tableint> result;
        nodeidx_t cur = current_root_;
        nodeidx_t max_level = nodes_[cur].level();
        for (auto elem : alive_elements_) {
            if (elem.second <= timestamp) {
                result.push_back(elem.first);
            }
        }

        for (nodeidx_t son_level = max_level - 1; son_level >= 0; son_level--) {
            assert(cur < nodes_.size());
            nodes_[cur].search(result, timestamp);
            if (nodes_[cur].split() == timestamp) {
                break;
            }
            else if (timestamp < nodes_[cur].split()) {
                cur -= (1 << son_level);
            }
            else {
                cur += (1 << son_level);
                while (cur >= nodes_.size()) {
                    son_level -= 1;
                    cur -= (1 << son_level);
                }
            }
        }

        nodes_[cur].search(result, timestamp);

        return result;
    }

    std::pair<size_t, size_t> memory_cost() const {
        size_t cost = sizeof(NeighborTree);
        for (const auto& node : nodes_) {
            cost += node.memory_cost();
        }
        return { cost, nodes_.size() };
    }
};

} // namespace timestampgraph