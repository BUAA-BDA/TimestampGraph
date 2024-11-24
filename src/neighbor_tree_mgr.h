#pragma once
#include "neighbor_mgr.h"
#include "neighbor_tree.h"

namespace timestampgraph {

class NeighborTreeMgr : public SpanMgr {

private:
    NeighborTree* trees0_;
    NeighborTree** trees_;
    std::vector<linklistsizeint> element_levels_;

    size_t node_size0_{ 0 }, node_size_{ 0 };

public:

    NeighborTreeMgr(size_t max_elements, size_t node_size, size_t node_size0)
        : node_size0_(node_size0), node_size_(node_size), SpanMgr(max_elements) {
        std::allocator<NeighborTree> alloc;
        trees0_ = alloc.allocate(max_elements);

        std::allocator<NeighborTree*> alloc_ptr;
        trees_ = alloc_ptr.allocate(max_elements);

        element_levels_.resize(max_elements);
        element_levels_.assign(max_elements, -1);
    }

    ~NeighborTreeMgr() {
        std::allocator<NeighborTree> alloc;
        alloc.deallocate(trees0_, max_elements_);

        for (size_t i = 0; i < max_elements_; i++) {
            if (element_levels_[i] > 0) {
                std::allocator<NeighborTree> alloc;
                for (int j = 0; j < element_levels_[i] - 1; j++) {
                    alloc.destroy(trees_[i] + j);
                }
                alloc.deallocate(trees_[i], element_levels_[i] - 1);
            }
        }

        std::allocator<NeighborTree*> alloc_ptr;
        alloc_ptr.deallocate(trees_, max_elements_);
    }

    inline const std::vector<tableint> get_linklist0(tableint internal_id, timestamp_t ts) const {
        return trees0_[internal_id].get_linklist(ts);
    }

    inline const std::vector<tableint> get_linklist1(tableint internal_id, int level, timestamp_t ts) const {
        auto ptr = trees_[internal_id][level - 1];
        return ptr.get_linklist(ts);
    }


    inline const std::vector<tableint> get_linklist(tableint internal_id, int level, timestamp_t ts) const {
        return level == 0 ? get_linklist0(internal_id, ts) : get_linklist1(internal_id, level, ts);
    }

    inline void alloc_linklist(tableint internal_id, timestamp_t ts, int level) {
        assert(element_levels_[internal_id] == -1);
        std::allocator<NeighborTree> alloc0;
        alloc0.construct(trees0_ + internal_id, node_size0_);
        set_birth(internal_id, ts);
        if (level > 0) {
            std::allocator<NeighborTree> alloc;
            trees_[internal_id] = alloc.allocate(level);

            for (int i = 0; i < level; i++) {
                alloc.construct(trees_[internal_id] + i, node_size_);
            }
        }
        element_levels_[internal_id] = level;
    }

    inline void check_valid(tableint internal_id, int level, timestamp_t ts, linklistsizeint expect_size = -1) const {
        auto ll = get_linklist(internal_id, level, ts);
        if (expect_size >= 0 && ll.size() != expect_size)
            throw std::runtime_error("check_valid: size error");
        for (tableint i : ll) {
            if (i < 0 || i > max_elements_)
                throw std::runtime_error("check_valid: neighbor error");
        }
    }

    inline linklistsizeint element_level(tableint internal_id) const {
        return element_levels_[internal_id];
    }

    inline void append_linklist(tableint internal_id, int level, tableint val, timestamp_t ts) {
        if (level == 0) {
            trees0_[internal_id].set_linklist(ts, val);
        }
        else {
            trees_[internal_id][level - 1].set_linklist(ts, val);
        }
    }

    inline void set_linklist(tableint internal_id, int level, const std::vector<tableint>& list, timestamp_t ts) {
        if (level == 0) {
            trees0_[internal_id].set_linklist(ts, list);
        }
        else {
            trees_[internal_id][level - 1].set_linklist(ts, list);
        }
    }

    inline size_t memory_cost() const {
        size_t struct_cost = SpanMgr::memory_cost() + sizeof(NeighborTreeMgr) - sizeof(SpanMgr);
        size_t vtcost = 0;
        size_t vtcount = 0;
        size_t vtnodes = 0;
        vtcost += max_elements_ * sizeof(NeighborTree);
        vtcount += max_elements_;
        for (size_t i = 0; i < max_elements_; i++) {
            auto pair = trees0_[i].memory_cost();
            vtcost += pair.first;
            vtnodes += pair.second;
            if (element_levels_[i] > 0) {
                vtcost += (element_levels_[i] - 1) * sizeof(NeighborTree);
                vtcount += element_levels_[i] - 1;
                for (int j = 0; j < element_levels_[i] - 1; j++) {
                    auto pair = trees_[i][j].memory_cost();
                    vtcost += pair.first;
                    vtnodes += pair.second;
                }
            }
        }
        struct_cost += max_elements_ * sizeof(linklistsizeint);
        std::cout << "struct_cost: " << struct_cost << std::endl;
        std::cout << "vtcost(" << vtcount << "vt with " << vtnodes << " nodes): " << vtcost << std::endl;
        return struct_cost + vtcost;
    }
};


class NeighborTreeMgrWithPossibleNeighbors : public NeighborTreeMgr {

private:

    PossibleNeighborsMgr<tableint> backup_;

public:

    NeighborTreeMgrWithPossibleNeighbors(size_t max_elements, size_t node_size, size_t node_size0)
        : NeighborTreeMgr(max_elements, node_size, node_size0), backup_(max_elements, node_size / 2) {}

    inline void set_possible_neighbor(tableint internal_id, tableint val) {
        backup_.set_possible_neighbor(internal_id, val);
    }

    inline void set_possible_neighbors(tableint internal_id, std::vector<tableint> vals) {
        backup_.set_possible_neighbors(internal_id, vals);
    }

    inline const std::vector<tableint> get_possible_neighbors(tableint internal_id, timestamp_t ts) {
        return backup_.get_possible_neighbors(internal_id, ts);
    }

    inline size_t memory_cost() const {
        size_t cost = NeighborTreeMgr::memory_cost();
        return cost;
    }
};

} // namespace timestampgraph