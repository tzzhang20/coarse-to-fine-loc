#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <nanoflann.hpp>
#include <utils/KDTreeVectorOfVectorsAdaptor.h>
#include <utils/serializer.h>

#include <xsearch.h>

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = point_type;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

class ISCManager
{
public:
    ISCManager() = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makeScancontext(const pcl::PointCloud<SCPointType> &_scan_down);
    Eigen::MatrixXd makeRingkeyFromScancontext(const Eigen::MatrixXd &_desc);
    Eigen::MatrixXd makeSectorkeyFromScancontext(const Eigen::MatrixXd &_desc);

    int fastAlignUsingVkey(const Eigen::MatrixXd &_vkey1, const Eigen::MatrixXd &_vkey2);
    double distDirectSC(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2);                           // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnScanContext(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSaveScancontextAndKeys(const Eigen::MatrixXd &);
    std::pair<int, float> detectLoopClosureID(const Eigen::MatrixXd &curr_desc, float *bias); // int: nearest node index, float: relative yaw

    size_t searchTopN(const Eigen::MatrixXd &query_desc, size_t max_result, search_result *results);
    void buildIndex();

public:
    // hyper parameters ()
    double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    int PC_NUM_RING = 20;        // 20 in the original paper (IROS 18)
    int PC_NUM_SECTOR = 60;      // 60 in the original paper (IROS 18)
    double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    int NUM_CANDIDATES_FROM_TREE = 10; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    double SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.

    // data
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;
};

static inline float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

static inline float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}

static inline float xy2theta(const float &_x, const float &_y)
{
    if (_x >= 0 & _y >= 0)
        return (180 / M_PI) * atan(_y / _x);

    if (_x < 0 & _y >= 0)
        return 180 - ((180 / M_PI) * atan(_y / (-_x)));

    if (_x < 0 & _y < 0)
        return 180 + ((180 / M_PI) * atan(_y / _x));

    if (_x >= 0 & _y < 0)
        return 360 - ((180 / M_PI) * atan((-_y) / _x));

    return 0.0f;
} // xy2theta

static MatrixXd circshift(const MatrixXd &_mat, int _num_shift)
{
    // shift columns to right direction
    assert(_num_shift >= 0);

    if (_num_shift == 0)
    {
        MatrixXd shifted_mat(_mat);
        return shifted_mat; // Early return
    }

    MatrixXd shifted_mat = MatrixXd::Zero(_mat.rows(), _mat.cols());
    for (int col_idx = 0; col_idx < _mat.cols(); col_idx++)
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;

} // circshift

static inline std::vector<float> eig2stdvec(const Eigen::MatrixXd &_eigmat)
{
    std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
    return vec;
} // eig2stdvec

double ISCManager::distDirectSC(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2)
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++)
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);

        if (col_sc1.norm() == 0 || col_sc2.norm() == 0)
            continue; // don't count this sector pair.

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

} // distDirectSC

int ISCManager::fastAlignUsingVkey(const MatrixXd &_vkey1, const MatrixXd &_vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++)
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm = vkey_diff.norm();
        if (cur_diff_norm < min_veky_diff_norm)
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey

std::pair<double, int> ISCManager::distanceBtnScanContext(const MatrixXd &_sc1, const MatrixXd &_sc2)
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

    const int SEARCH_RADIUS = round(0.5 * SEARCH_RATIO * _sc1.cols()); // a half of search range
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++)
    {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. fast columnwise diff
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for (int num_shift : shift_idx_search_space)
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
        if (cur_sc_dist < min_sc_dist)
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

MatrixXd ISCManager::makeScancontext(const pcl::PointCloud<SCPointType> &_scan_down)
{
    int num_pts_scan_down = _scan_down.points.size();

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sctor_idx;

    for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_down.points[pt_idx].x;
        pt.y = _scan_down.points[pt_idx].y;
        pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).
        pt.intensity = _scan_down.points[pt_idx].intensity;
        // xyz to ring, sector
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);

        // if range is out of roi, pass
        if (azim_range > PC_MAX_RADIUS)
            continue;

        ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
        sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), 1);

        // taking maximum z
        if (desc(ring_idx - 1, sctor_idx - 1) < pt.intensity) // -1 means cpp starts from 0
            desc(ring_idx - 1, sctor_idx - 1) = pt.intensity; // update for taking maximum value at that bin
    }

    // reset no points to zero (for cosine dist later)
    for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
        for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
            if (desc(row_idx, col_idx) == NO_POINT)
                desc(row_idx, col_idx) = 0;

    return desc;
} // ISCManager::makeScancontext

MatrixXd ISCManager::makeRingkeyFromScancontext(const Eigen::MatrixXd &_desc)
{
    /*
     * summary: rowwise mean vector
     */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for (int row_idx = 0; row_idx < _desc.rows(); row_idx++)
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }

    return invariant_key;
} // ISCManager::makeRingkeyFromScancontext

MatrixXd ISCManager::makeSectorkeyFromScancontext(const Eigen::MatrixXd &_desc)
{
    /*
     * summary: columnwise mean vector
     */
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for (int col_idx = 0; col_idx < _desc.cols(); col_idx++)
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }

    return variant_key;
} // ISCManager::makeSectorkeyFromScancontext

void ISCManager::makeAndSaveScancontextAndKeys(const Eigen::MatrixXd &sc)
{
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);

    // cout <<polarcontext_vkeys_.size() << endl;

} // ISCManager::makeAndSaveScancontextAndKeys

std::pair<int, float> ISCManager::detectLoopClosureID(const Eigen::MatrixXd &curr_desc, float *bias)
{
    int loop_id{-1}; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")
    auto ring_key = makeRingkeyFromScancontext(curr_desc);
    auto curr_key = eig2stdvec(ring_key); // current observation (query)

    /*
     * step 1: candidates from ringkey tree_
     */
    if (polarcontext_invkeys_to_search_.empty())
    {
        std::pair<int, float> result{loop_id, 0.0};
        return result; // Early return
    }

    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
    knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10));

    /*
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    for (int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++)
    {
        MatrixXd polarcontext_candidate = polarcontexts_[candidate_indexes[candidate_iter_idx]];
        std::pair<double, int> sc_dist_result = distanceBtnScanContext(curr_desc, polarcontext_candidate);

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if (candidate_dist < min_dist)
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }

    std::pair<int, float> result{nn_idx, min_dist};
    if (bias != nullptr)
        *bias = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    return result;
} // ISCManager::detectLoopClosureID

size_t ISCManager::searchTopN(const Eigen::MatrixXd &query_desc, size_t max_result, search_result *results)
{
    auto ring_key = makeRingkeyFromScancontext(query_desc);
    auto curr_key = eig2stdvec(ring_key); // current observation (query)

    /*
     * step 1: candidates from ringkey tree_
     */
    if (polarcontext_invkeys_to_search_.size() < max_result)
    {
        max_result = polarcontext_invkeys_to_search_.size();
    }

    if (max_result == 0)
    {
        return 0;
    }

    size_t CANDIDATES = max_result < 5 ? 10 : max_result * 2;
    if (CANDIDATES > polarcontext_invkeys_to_search_.size())
    {
        CANDIDATES = polarcontext_invkeys_to_search_.size();
    }

    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indexes(CANDIDATES);
    std::vector<float> out_dists_sqr(CANDIDATES);

    nanoflann::KNNResultSet<float> knnsearch_result(CANDIDATES);
    knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10));

    std::vector<search_result> search_results(CANDIDATES);
    /*
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    for (int candidate_iter_idx = 0; candidate_iter_idx < CANDIDATES; candidate_iter_idx++)
    {
        MatrixXd polarcontext_candidate = polarcontexts_[candidate_indexes[candidate_iter_idx]];
        std::pair<double, int> sc_dist_result = distanceBtnScanContext(query_desc, polarcontext_candidate);

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        search_results[candidate_iter_idx].id = candidate_indexes[candidate_iter_idx];
        search_results[candidate_iter_idx].score = candidate_dist;
        search_results[candidate_iter_idx].yaw = deg2rad(candidate_align * PC_UNIT_SECTORANGLE);
    }

    std::sort(search_results.begin(), search_results.end(), [](const search_result &a, const search_result &b)
              { return a.score < b.score; });

    for (size_t i = 0; i < max_result; i++)
    {
        results[i] = search_results[i];
    }

    return max_result;
} // ISCManager::searchTopN
void ISCManager::buildIndex()
{
    polarcontext_invkeys_to_search_.clear();
    polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end());

    polarcontext_tree_.reset();
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);
}

// SC module
#include <xsearch.h>

constexpr const char *iscancontext_module_name = "iscancontext";

struct iscancontext_device : search_device
{
    ISCManager isc_manager;
    std::vector<MatrixXd> object_values;
    std::vector<object_id> joined_object_id;
};

static object_id __isc_create_object(search_device *object, const pcl::PointCloud<point_type> &cloud)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return obj_none;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    auto matrix = sc_object->isc_manager.makeScancontext(cloud);
    sc_object->object_values.push_back(matrix);
    return sc_object->object_values.size() - 1;
}

static size_t __isc_search(search_device *object, object_id searched_target, search_result *results, size_t max_results)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return 0;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);

    if (searched_target >= sc_object->object_values.size())
    {
        return 0;
    }

    return sc_object->isc_manager.searchTopN(sc_object->object_values[searched_target], max_results, results);
}

static bool __isc_config(search_device *object, const char *key, const char *value)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return false;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    if (strcmp(key, "lidar_height") == 0)
    {
        sc_object->isc_manager.LIDAR_HEIGHT = std::stod(value);
        return true;
    }
    if (strcmp(key, "num_ring") == 0)
    {
        sc_object->isc_manager.PC_NUM_RING = std::stoi(value);
        sc_object->isc_manager.PC_UNIT_RINGGAP = sc_object->isc_manager.PC_MAX_RADIUS / double(sc_object->isc_manager.PC_NUM_RING);
        return true;
    }
    if (strcmp(key, "num_sector") == 0)
    {
        sc_object->isc_manager.PC_NUM_SECTOR = std::stoi(value);
        sc_object->isc_manager.PC_UNIT_SECTORANGLE = 360.0 / double(sc_object->isc_manager.PC_NUM_SECTOR);
        return true;
    }
    if (strcmp(key, "max_radius") == 0)
    {
        sc_object->isc_manager.PC_MAX_RADIUS = std::stod(value);
        sc_object->isc_manager.PC_UNIT_RINGGAP = sc_object->isc_manager.PC_MAX_RADIUS / double(sc_object->isc_manager.PC_NUM_RING);
        return true;
    }

    if (strcmp(key, "num_candidates") == 0)
    {
        sc_object->isc_manager.NUM_CANDIDATES_FROM_TREE = std::stof(value);
        return true;
    }

    if (strcmp(key, "ratio") == 0)
    {
        sc_object->isc_manager.SEARCH_RATIO = std::stod(value);
        return true;
    }

    return false;
}

static void __isc_join(search_device *object, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    sc_object->joined_object_id.push_back(id);
    sc_object->isc_manager.makeAndSaveScancontextAndKeys(sc_object->object_values[id]);
}

static void __isc_join_flush(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    sc_object->isc_manager.buildIndex();
}

static ssize_t __isc_serialize(search_device *object, FILE *fp, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return -1;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    if (id >= sc_object->object_values.size())
    {
        return -1;
    }

    return serialize_eigen(fp, sc_object->object_values[id]);
}

static ssize_t __isc_deserialize(search_device *object, FILE *fp, object_id &id)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return -1;
    }

    auto sc_object = static_cast<iscancontext_device *>(object);
    if (id >= sc_object->object_values.size())
    {
        return -1;
    }
    Eigen::MatrixXd matrix;
    auto result = deserialize_eigen(fp, matrix);
    if (result < 0)
    {
        return result;
    }
    sc_object->object_values.push_back(matrix);
    id = sc_object->object_values.size() - 1;
    return result;
}

static search_device *__isc_create();
static void __isc_destroy(search_device *object);

static search_module isc_module = {
    .name = iscancontext_module_name, // The module name.
    .create_object = &__isc_create_object,
    .search = &__isc_search,
    .config = &__isc_config,
    .join = &__isc_join,
    .join_flush = &__isc_join_flush,
    .serialize = &__isc_serialize,
    .deserialize = &__isc_deserialize,
    .save = nullptr,
    .create = &__isc_create,
    .destroy = &__isc_destroy,
};

static search_device *__isc_create()
{
    auto object = new iscancontext_device;
    object->module = &isc_module;
    return object;
}

static void __isc_destroy(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, iscancontext_module_name) != 0)
    {
        return;
    }

    delete static_cast<iscancontext_device *>(object);
}

register_search_module(isc_module);
