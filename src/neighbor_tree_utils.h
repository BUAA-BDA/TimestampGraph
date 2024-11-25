#include <vector>
#include <stdexcept>
#include <cassert>

namespace timestampgraph {

typedef int timestamp_t;
typedef std::pair<timestamp_t, timestamp_t> interval_t;
typedef short nodeidx_t;

class NeighborTreeUtil {
    static constexpr size_t MAX_LEVEL = 1000;
    struct {
        nodeidx_t level;
        nodeidx_t parent;
    } idx_info[MAX_LEVEL];
    
    inline nodeidx_t _calc_level(nodeidx_t idx) {
        nodeidx_t level = 0;
        while ((idx + 1) / (1 << level) * (1 << level) == idx + 1) {
            level += 1;
        }
        return level - 1;
    }

    inline nodeidx_t _calc_parent_with_cache(nodeidx_t idx) {
        nodeidx_t level = idx_info[idx].level;
        
        nodeidx_t rcand = idx + (1 << level);
        nodeidx_t rlevel = rcand < MAX_LEVEL ? idx_info[rcand].level : _calc_level(rcand);
        if (rlevel == level + 1) {
            return rcand;
        }

        nodeidx_t lcand = idx - (1 << level);
        nodeidx_t llevel = idx_info[lcand].level;
        if (llevel == level + 1) {
            return lcand;
        }

        throw std::runtime_error("Invalid parent");
    }

    inline nodeidx_t _calc_parent(nodeidx_t idx) {
        nodeidx_t level = _calc_level(idx);
        
        nodeidx_t rcand = idx + (1 << level);
        if (_calc_level(rcand) == level + 1) {
            return rcand;
        }

        nodeidx_t lcand = idx - (1 << level);
        if (_calc_level(lcand) == level + 1) {
            return lcand;
        }

        throw std::runtime_error("Invalid parent");
    }




    void init_idx_info() {
        for (nodeidx_t i = 0; i < MAX_LEVEL; i++) {
            idx_info[i].level = _calc_level(i);
        }
        for (nodeidx_t i = 0; i < MAX_LEVEL; i++) {
            idx_info[i].parent = _calc_parent_with_cache(i);
        }
    }

public:

    NeighborTreeUtil() {
        init_idx_info();
    }

    nodeidx_t calc_level(nodeidx_t idx) {
        return idx < MAX_LEVEL ? idx_info[idx].level : _calc_level(idx);
    }

    nodeidx_t calc_parent(nodeidx_t idx) {
        return idx < MAX_LEVEL ? idx_info[idx].parent : _calc_parent(idx);
    }
};

NeighborTreeUtil _neighbor_tree_util;

nodeidx_t calc_level(nodeidx_t idx) {
    return _neighbor_tree_util.calc_level(idx);
}

nodeidx_t calc_parent(nodeidx_t idx) {
    return _neighbor_tree_util.calc_parent(idx);
}

} // namespace timestampgraph