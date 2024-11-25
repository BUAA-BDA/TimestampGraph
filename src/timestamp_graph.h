#pragma once
#include <hnswlib/hnswlib.h>
#include <interface.h>
#include "neighbor_tree_mgr.h"

namespace timestampgraph {

using hnswlib::DISTFUNC;
using hnswlib::labeltype;
using hnswlib::SpaceInterface;
using hnswlib::VisitedList;
using hnswlib::VisitedListPool;
using hnswlib::vl_type;

template <typename dist_t, typename nodeid_t, typename vtmgr_t>
class TimestampGraphBase {
public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;
    static constexpr float LEVEL_GENARATOR_LOWER_BOUND = 0.001;

    size_t max_elements_{ 0 };
    mutable std::atomic<size_t> cur_element_count{ 0 };

    size_t M_{ 0 };
    size_t maxM_{ 0 };
    size_t maxM0_{ 0 };
    size_t ef_construction_{ 0 };
    size_t ef_{ 0 };

    double mult_{ 0.0 }, revSize_{ 0.0 };
    int maxlevel_{ 0 };

    std::unique_ptr<VisitedListPool> visited_list_pool_{ nullptr };

    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    mutable std::vector<std::mutex> link_list_locks_;

    nodeid_t enterpoint_node_{ 0 };

    DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_{ nullptr };

    mutable std::mutex label_lookup_lock;

    std::unordered_map<labeltype, nodeid_t> label_lookup_;

    std::default_random_engine level_generator_;

private:
    char* data_level0_memory_{ nullptr };

    size_t size_element_{ 0 };

    size_t size_data_{ 0 };

    size_t offset_data_{ 0 }, offset_label_{ 0 };

    vtmgr_t vtmgr_;

    constexpr char* get_data_ptr(nodeid_t internal_id) const {
        return data_level0_memory_ + internal_id * size_element_ + offset_data_;
    }

    constexpr labeltype* get_label_ptr(nodeid_t internal_id) const {
        return (labeltype*) (data_level0_memory_ + internal_id * size_element_ + offset_label_);
    }

    inline bool search_neighbors(const void* data_point, const std::vector<nodeid_t>& ll, nodeid_t& nn_id, dist_t& min_dist) const {
        bool changed = false;
        for (nodeid_t cand : ll) {
            if (cand < 0 || cand > max_elements_)
                throw std::runtime_error("search_neighbors: cand error");
            dist_t d = fstdistfunc_(data_point, get_data_ptr(cand), dist_func_param_);
            if (d < min_dist) {
                min_dist = d;
                nn_id = cand;
                changed = true;
            }
        }
        return changed;
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, nodeid_t> const& a,
            std::pair<dist_t, nodeid_t> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    template <bool use_lock = false>
    inline nodeid_t search_downstep(const void* data_point, nodeid_t ep_id,
        int top_level, int stop_level, timestamp_t ts) const {
        nodeid_t cur_obj = ep_id;
        dist_t cur_dist = fstdistfunc_(data_point, get_data_ptr(cur_obj), dist_func_param_);
        for (int level = top_level; level > stop_level; level--) {
            bool changed = true;
            while (changed) {
                std::unique_lock<std::mutex> lock;
                if (use_lock) {
                    lock = std::unique_lock<std::mutex>(link_list_locks_[cur_obj]);
                }
                changed = search_neighbors(data_point, vtmgr_.get_linklist(cur_obj, level, ts), cur_obj, cur_dist);
            }
        }
        return cur_obj;
    }

    template <typename container_t>
    inline auto select_nearest_neighbors(container_t& candidates, nodeid_t src, size_t N) const
        -> std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst> {

        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst> result;

        for (nodeid_t cand : candidates) {
            if (cand == src)
                continue;

            dist_t distance = fstdistfunc_(get_data_ptr(src), get_data_ptr(cand), dist_func_param_);
            if (result.size() < N) {
                result.emplace(distance, cand);
            }
            else {
                if (distance < result.top().first) {
                    result.pop();
                    result.emplace(distance, cand);
                }
            }
        }

        return result;
    }

public:
    TimestampGraphBase(
        SpaceInterface<dist_t>* s,
        size_t max_elements,
        size_t M,
        size_t vt_size,
        size_t ef_construction,
        size_t random_seed = 100)
        : max_elements_(max_elements), label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
        link_list_locks_(max_elements),
        M_(M <= 10000 ? M : 10000), maxM_(M_), maxM0_(M_ * 2),
        vtmgr_(max_elements, vt_size, vt_size * 2) {
        size_data_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (M > 10000) {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
        }

        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);

        size_element_ = size_data_ + sizeof(labeltype);

        offset_data_ = 0;
        offset_label_ = 0 + size_data_;

        data_level0_memory_ = (char*) malloc(max_elements_ * size_element_);

        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;
        maxlevel_ = -1;

        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~TimestampGraphBase() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;

        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }

    void setEf(size_t ef) {
        ef_ = ef;
    }

    inline std::mutex& getLabelOpMutex(labeltype label) const {
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(LEVEL_GENARATOR_LOWER_BOUND, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    template <bool use_lock = false, bool base_layer = true>
    std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>
        searchBaseLayerST(
            nodeid_t ep_id,
            const void* data_point,
            size_t ef,
            timestamp_t ts,
            int level = 0) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        dist_t dist = fstdistfunc_(data_point, get_data_ptr(ep_id), dist_func_param_);

        if (vtmgr_.check_alive(ep_id, ts)) {
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
        }
        else {
            lowerBound = std::numeric_limits<dist_t>::max();
        }
        candidate_set.emplace(-dist, ep_id);

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, nodeid_t> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            nodeid_t current_node_id = current_node_pair.second;

            std::unique_lock<std::mutex> lock;
            if (use_lock) {
                lock = std::unique_lock<std::mutex>(link_list_locks_[current_node_id]);
            }

            std::vector<nodeid_t> ll;
            if (base_layer) {
                ll = vtmgr_.get_linklist0(current_node_id, ts);
            }
            else {
                ll = vtmgr_.get_linklist(current_node_id, level, ts);
            }

            for (nodeid_t candidate_id : ll) {
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char* currObj1 = (get_data_ptr(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    bool is_alive = vtmgr_.check_alive(candidate_id, ts);

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
                        if (is_alive) {
                            top_candidates.emplace(dist, candidate_id);
                        }

                        while (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <bool backup = false>
    void getNeighborsByHeuristic2(
        nodeid_t internal_id,
        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>& top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, nodeid_t>> queue_closest;
        std::vector<std::pair<dist_t, nodeid_t>> return_list;

        std::vector<nodeid_t> backup_list;

        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, nodeid_t> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, nodeid_t> second_pair : return_list) {
                dist_t curdist =
                    fstdistfunc_(get_data_ptr(second_pair.second),
                        get_data_ptr(curent_pair.second),
                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }

            if (good) {
                return_list.push_back(curent_pair);
            }
            else if (backup && backup_list.size() < M) {
                backup_list.push_back(curent_pair.second);
            }
        }

        if (backup) {
            vtmgr_.set_possible_neighbors(internal_id, backup_list);
        }

        for (std::pair<dist_t, nodeid_t> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    nodeid_t mutuallyConnectNewElement(
        const void* data_point,
        nodeid_t cur_c,
        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>& top_candidates,
        int level,
        bool isUpdate,
        timestamp_t ts) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2<true>(cur_c, top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<nodeid_t> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        nodeid_t next_closest_entry_point = selectedNeighbors.back();

        {

            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }

            for (nodeid_t neigh : selectedNeighbors) {
                if (level > vtmgr_.element_level(neigh))
                    throw std::runtime_error("Trying to make a link on a non-existent level");
            }

            vtmgr_.check_valid(cur_c, level, ts, isUpdate ? -1 : 0);
            vtmgr_.set_linklist(cur_c, level, selectedNeighbors, ts);
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            std::vector<nodeid_t> ll_other = vtmgr_.get_linklist(selectedNeighbors[idx], level, ts);

            size_t sz_link_list_other = ll_other.size();

            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > vtmgr_.element_level(selectedNeighbors[idx]))
                throw std::runtime_error("Trying to make a link on a non-existent level");

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (nodeid_t neigh : ll_other) {
                    if (neigh == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    vtmgr_.append_linklist(selectedNeighbors[idx], level, cur_c, ts);
                }
                else {

                    dist_t d_max = fstdistfunc_(get_data_ptr(cur_c), get_data_ptr(selectedNeighbors[idx]),
                        dist_func_param_);

                    std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (nodeid_t neigh : ll_other) {
                        dist_t d = fstdistfunc_(get_data_ptr(neigh), get_data_ptr(selectedNeighbors[idx]), dist_func_param_);
                        candidates.emplace(d, neigh);
                    }

                    getNeighborsByHeuristic2(selectedNeighbors[idx], candidates, Mcurmax);

                    std::vector<nodeid_t> candidates_list;
                    while (candidates.size() > 0) {
                        candidates_list.push_back(candidates.top().second);
                        candidates.pop();
                    }

                    vtmgr_.set_linklist(selectedNeighbors[idx], level, candidates_list, ts);
                }
            }
        }

        return next_closest_entry_point;
    }

    void markDelete(labeltype label, timestamp_t ts) {

        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        nodeid_t internalId = search->second;
        lock_table.unlock();

        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId, ts)) {
            vtmgr_.set_death(internalId, ts);
            updatePoint(internalId, ts);
        }
        else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    bool isMarkedDeleted(nodeid_t internalId, timestamp_t ts) const {
        return !vtmgr_.check_alive(internalId, ts);
    }

    void addPoint(const void* data_point, labeltype label, timestamp_t ts) {

        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        addPoint(data_point, label, -1, ts);
    }

    void updatePoint(nodeid_t dead_point_id, timestamp_t ts) {

        int maxLevelCopy = maxlevel_;
        nodeid_t entryPointCopy = enterpoint_node_;

        if (entryPointCopy == dead_point_id && cur_element_count == 1)
            return;

        int elemLevel = vtmgr_.element_level(dead_point_id);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<nodeid_t> sCand;
            std::unordered_set<nodeid_t> sNeigh;

            bool need_search = false;

            std::vector<nodeid_t> listOneHop;
            {
                std::unique_lock<std::mutex> lock(link_list_locks_[dead_point_id]);
                listOneHop = vtmgr_.get_linklist(dead_point_id, layer, ts);
            }
            if (listOneHop.size() == 0)
                continue;

            for (auto&& elOneHop : listOneHop) {

                sNeigh.emplace(elOneHop);

                if (vtmgr_.check_alive(elOneHop, ts))
                    sCand.emplace(elOneHop);

                std::unique_lock<std::mutex> lock(link_list_locks_[elOneHop]);
                std::vector<nodeid_t> elTwoHops = vtmgr_.get_linklist(elOneHop, layer, ts);
                for (nodeid_t elTwoHop : elTwoHops) {
                    if (vtmgr_.check_alive(elTwoHop, ts))
                        sCand.emplace(elTwoHop);
                }

                auto elTwoHopsBackup = vtmgr_.get_possible_neighbors(elOneHop, ts);
                if (elTwoHopsBackup.size() == 0) {
                    need_search = true;
                }
                else
                    for (nodeid_t elTwoHop : elTwoHopsBackup) {
                        if (vtmgr_.check_alive(elTwoHop, ts) && vtmgr_.element_level(elTwoHop) >= layer)
                            sCand.emplace(elTwoHop);
                    }
            }

            if (sCand.size() == 0 || need_search) {
                auto res = searchBaseLayerST<true, false>(entryPointCopy, get_data_ptr(dead_point_id), ef_construction_, ts, layer);
                if (res.size() == 0) {
                    continue;
                }
            }

            if (sCand.find(dead_point_id) != sCand.end())
                throw std::runtime_error("Find Self Connection When Updating");

            for (nodeid_t neigh : sNeigh) {
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;
                size_t elementsToKeep = std::min(ef_construction_, size);
                std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>
                    candidates = select_nearest_neighbors(sCand, neigh, elementsToKeep);

                getNeighborsByHeuristic2(neigh, candidates, layer == 0 ? maxM0_ : maxM_);

                std::vector<nodeid_t> candidates_list;
                while (candidates.size() > 0) {
                    candidates_list.push_back(candidates.top().second);
                    candidates.pop();
                }

                std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                vtmgr_.set_linklist(neigh, layer, candidates_list, ts);
            }
        }
    }

    nodeid_t addPoint(const void* data_point, labeltype label, int level, timestamp_t ts) {
        nodeid_t cur_c = 0;
        {

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);

            if (search != label_lookup_.end()) {
                throw std::runtime_error("Update is not supported yet");
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;
        if (enterpoint_node_ == -1)
            curlevel = (int) (-log(LEVEL_GENARATOR_LOWER_BOUND) * mult_);

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();

        nodeid_t currObj = enterpoint_node_;
        nodeid_t enterpoint_copy = enterpoint_node_;

        vtmgr_.alloc_linklist(cur_c, ts, curlevel);

        memcpy(get_label_ptr(cur_c), &label, sizeof(labeltype));
        memcpy(get_data_ptr(cur_c), data_point, size_data_);

        if ((signed) currObj != -1) {
            if (curlevel < maxlevelcopy) {
                currObj = search_downstep(data_point, currObj, maxlevelcopy, curlevel, ts);
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy, ts);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>
                    top_candidates = searchBaseLayerST<true, false>(currObj, data_point, ef_construction_, ts, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, get_data_ptr(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, ts);
            }
        }
        else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy && enterpoint_copy != -1) {
            throw std::runtime_error("unexpected level");
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void* query_data, size_t k, timestamp_t ts, size_t ef = 10) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        nodeid_t currObj = search_downstep(query_data, enterpoint_node_, maxlevel_, 0, ts);

        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, CompareByFirst>
            top_candidates = searchBaseLayerST(currObj, query_data, std::max(ef, k), ts);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, nodeid_t> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, *get_label_ptr(rez.second)));
            top_candidates.pop();
        }
        return result;
    }
};

template <typename dist_t, typename nodeid_t, typename vtmgr_t>
class TimestampGraphWrapper : public TANNSAlgorithmInterface<dist_t, labeltype> {

private:
    TimestampGraphBase<dist_t, nodeid_t, vtmgr_t>* index_;
    size_t operation_count_;
    bool normalize_;
    size_t dim_;

public:
    TimestampGraphWrapper(TANNSDataset& dataset, size_t M, size_t ef_construction, size_t vt_size, Metric metric) {
        if (metric == L2) {
            hnswlib::L2Space* space = new hnswlib::L2Space(dataset.dim_);
            index_ = new TimestampGraphBase<dist_t, nodeid_t, vtmgr_t>(space, dataset.n_elements_, M, vt_size, ef_construction);
            normalize_ = false;
        }
        else {
            hnswlib::InnerProductSpace* space = new hnswlib::InnerProductSpace(dataset.dim_);
            index_ = new TimestampGraphBase<dist_t, nodeid_t, vtmgr_t>(space, dataset.n_elements_, M, vt_size, ef_construction);
            normalize_ = true;
        }
        dim_ = dataset.dim_;
        operation_count_ = 0;
    }

    void addPoint(const void* datapoint, hnswlib::labeltype label) override {
        const float* data = (const float*) datapoint;
        if (normalize_) {
            data = copy_and_normalize_l2(data, dim_);
        }
        index_->addPoint(data, label, operation_count_++);
        if (normalize_) {
            delete[] data;
        }
    }

    void removePoint(hnswlib::labeltype label) {
        index_->markDelete(label, operation_count_++);
    }

    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
        searchKnn(const void* query_data, size_t k, size_t timestamp, size_t ef = 10) override {
        const float* query_data_normalized = (const float*) query_data;
        if (normalize_) {
            query_data_normalized = copy_and_normalize_l2((float*) query_data_normalized, dim_);
        }
        auto res = index_->searchKnn(query_data_normalized, k, timestamp, ef);
        if (normalize_) {
            delete[] query_data_normalized;
        }
        return res;
    }
};

template <typename dist_t>
class CompressedTimestampGraph : public TimestampGraphWrapper<dist_t, tableint, NeighborTreeMgrWithPossibleNeighbors> {
public:
    CompressedTimestampGraph(TANNSDataset& dataset, size_t M, size_t ef_construction, size_t vt_size, Metric metric)
        : TimestampGraphWrapper<dist_t, tableint, NeighborTreeMgrWithPossibleNeighbors>(dataset, M, ef_construction, vt_size, metric) {}

    const char* name() override {
        return "CompressedTimestampGraph";
    }
};

template <typename dist_t>
class TimestampGraph : public TimestampGraphWrapper<dist_t, size_t, NeighborsMgrWithPossibleNeighbors<size_t>> {
public:
    TimestampGraph(TANNSDataset& dataset, size_t M, size_t ef_construction, size_t vt_size, Metric metric)
        : TimestampGraphWrapper<dist_t, size_t, NeighborsMgrWithPossibleNeighbors<size_t>>(dataset, M, ef_construction, vt_size, metric) {}

    const char* name() override {
        return "TimestampGraph";
    }
};

} // namespace timestampgraph