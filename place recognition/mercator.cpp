#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <fstream>
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

class MSCManager
{
public:
    MSCManager() = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makemercator(const pcl::PointCloud<SCPointType> &_scan_down);
    Eigen::MatrixXd makeRingkeyFrommercator(const Eigen::MatrixXd &_desc);
    Eigen::MatrixXd makeSectorkeyFrommercator(const Eigen::MatrixXd &_desc);

    int fastAlignUsingVkey(const Eigen::MatrixXd &_vkey1, const Eigen::MatrixXd &_vkey2);
    double distDirectSC(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2);                           // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnmercator(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSavemercatorAndKeys(const Eigen::MatrixXd &);
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
    // const double SC_DIST_THRES = 0.5; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

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

double MSCManager::distDirectSC(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2)
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++)
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);

        if (col_sc1.norm() == 0 | col_sc2.norm() == 0)
            continue; // don't count this sector pair.

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

} // distDirectSC

int MSCManager::fastAlignUsingVkey(const Eigen::MatrixXd &_vkey1, const Eigen::MatrixXd &_vkey2)
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

std::pair<double, int> MSCManager::distanceBtnmercator(const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2)
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = makeSectorkeyFrommercator(_sc1);
    MatrixXd vkey_sc2 = makeSectorkeyFrommercator(_sc2);
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

} // distanceBtnmercator

MatrixXd MSCManager::makemercator(const pcl::PointCloud<SCPointType> &_scan_down)
{
    // 使用两个矩阵，一个用于累积深度值，另一个用于计数
    MatrixXd descSum = MatrixXd::Zero(PC_NUM_RING, PC_NUM_SECTOR);  // 存储深度总和
    MatrixXi count = MatrixXi::Zero(PC_NUM_RING, PC_NUM_SECTOR);    // 存储每个区域的点数

    for (const auto& pt : _scan_down) {
        float range = std::hypot(pt.x, pt.y);
        if (range > PC_MAX_RADIUS || range < 2) continue;

        float angle = xy2theta(pt.x, pt.y);
        float angle_z = xy2theta(range, (pt.z+LIDAR_HEIGHT));

        int sectorIndex = std::clamp(int((angle / 360.0) * PC_NUM_SECTOR), 0, PC_NUM_SECTOR - 1);
        int ringIndex = std::clamp(int((angle_z / 45.0) * PC_NUM_RING), 0, PC_NUM_RING - 1);

        // 累加当前点的深度值到对应区域，并增加计数
        descSum(ringIndex, sectorIndex) += pt.z;
        count(ringIndex, sectorIndex) += 1;
    }

    // 计算平均深度值
    MatrixXd desc = MatrixXd::Zero(PC_NUM_RING, PC_NUM_SECTOR);
    for (int i = 0; i < PC_NUM_RING; ++i) {
        for (int j = 0; j < PC_NUM_SECTOR; ++j) {
            if (count(i, j) > 0) {
                desc(i, j) = descSum(i, j) / count(i, j);
            } else {
                desc(i, j) = 0; // 如果没有点落在某个区域，可以设为NO_POINT或者其他表示无效的值
            }
        }
    }

    return desc;
} // MSCManager::makemercator

MatrixXd MSCManager::makeRingkeyFrommercator(const Eigen::MatrixXd &_desc)
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
} // MSCManager::makeRingkeyFrommercator

MatrixXd MSCManager::makeSectorkeyFrommercator(const Eigen::MatrixXd &_desc)
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
} // MSCManager::makeSectorkeyFrommercator

void MSCManager::makeAndSavemercatorAndKeys(const Eigen::MatrixXd &sc)
{
    Eigen::MatrixXd ringkey = makeRingkeyFrommercator(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFrommercator(sc);
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);

    // cout <<polarcontext_vkeys_.size() << endl;

} // MSCManager::makeAndSavemercatorAndKeys

std::pair<int, float> MSCManager::detectLoopClosureID(const Eigen::MatrixXd &curr_desc, float *bias)
{
    int loop_id{-1}; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")
    auto ring_key = makeRingkeyFrommercator(curr_desc);
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
        std::pair<double, int> sc_dist_result = distanceBtnmercator(curr_desc, polarcontext_candidate);

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
}

size_t MSCManager::searchTopN(const Eigen::MatrixXd &query_desc, size_t max_result, search_result *results)
{
    auto ring_key = makeRingkeyFrommercator(query_desc);
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
        std::pair<double, int> sc_dist_result = distanceBtnmercator(query_desc, polarcontext_candidate);

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
} // IMSCManager::searchTopN

void MSCManager::buildIndex()
{
    polarcontext_invkeys_to_search_.clear();
    polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end());

    polarcontext_tree_.reset();
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);
}

// SC module
#include <xsearch.h>

constexpr const char *mercator_module_name = "mercator";

struct mercator_device : search_device
{
    MSCManager sc_manager;
    std::vector<MatrixXd> object_values;
    std::vector<object_id> joined_object_id;
};

static object_id __msc_create_object(search_device *object, const pcl::PointCloud<point_type> &cloud)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return obj_none;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    auto matrix = msc_object->sc_manager.makemercator(cloud);
    msc_object->object_values.push_back(matrix);
    return msc_object->object_values.size() - 1;
}

static size_t __msc_search(search_device *object, object_id searched_target, search_result *results, size_t max_results)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return 0;
    }

    auto msc_object = static_cast<mercator_device *>(object);

    if (searched_target >= msc_object->object_values.size())
    {
        return 0;
    }
    return msc_object->sc_manager.searchTopN(msc_object->object_values[searched_target], max_results, results);
}

static bool __msc_config(search_device *object, const char *key, const char *value)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return false;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    if (strcmp(key, "lidar_height") == 0)
    {
        msc_object->sc_manager.LIDAR_HEIGHT = std::stod(value);
        return true;
    }
    if (strcmp(key, "num_ring") == 0)
    {
        msc_object->sc_manager.PC_NUM_RING = std::stoi(value);
        msc_object->sc_manager.PC_UNIT_RINGGAP = msc_object->sc_manager.PC_MAX_RADIUS / double(msc_object->sc_manager.PC_NUM_RING);
        return true;
    }
    if (strcmp(key, "num_sector") == 0)
    {
        msc_object->sc_manager.PC_NUM_SECTOR = std::stoi(value);
        msc_object->sc_manager.PC_UNIT_SECTORANGLE = 360.0 / double(msc_object->sc_manager.PC_NUM_SECTOR);
        return true;
    }
    if (strcmp(key, "max_radius") == 0)
    {
        msc_object->sc_manager.PC_MAX_RADIUS = std::stod(value);
        msc_object->sc_manager.PC_UNIT_RINGGAP = msc_object->sc_manager.PC_MAX_RADIUS / double(msc_object->sc_manager.PC_NUM_RING);
        return true;
    }

    if (strcmp(key, "num_candidates") == 0)
    {
        msc_object->sc_manager.NUM_CANDIDATES_FROM_TREE = std::stof(value);
        return true;
    }

    if (strcmp(key, "ratio") == 0)
    {
        msc_object->sc_manager.SEARCH_RATIO = std::stod(value);
        return true;
    }

    return false;
}

static void __msc_join(search_device *object, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    msc_object->joined_object_id.push_back(id);
    msc_object->sc_manager.makeAndSavemercatorAndKeys(msc_object->object_values[id]);
}

static void __msc_join_flush(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    msc_object->sc_manager.buildIndex();
}

static ssize_t __msc_serialize(search_device *object, FILE *fp, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return -1;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    if (id >= msc_object->object_values.size())
    {
        return -1;
    }

    return serialize_eigen(fp, msc_object->object_values[id]);
}

static ssize_t __msc_deserialize(search_device *object, FILE *fp, object_id &id)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return -1;
    }

    auto msc_object = static_cast<mercator_device *>(object);
    if (id >= msc_object->object_values.size())
    {
        return -1;
    }
    Eigen::MatrixXd matrix;
    auto result = deserialize_eigen(fp, matrix);
    if (result < 0)
    {
        return result;
    }
    msc_object->object_values.push_back(matrix);
    id = msc_object->object_values.size() - 1;
    return result;
}

static bool __msc_save(search_device *object, object_id id, const char *filename)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return false;
    }
    auto msc_object = static_cast<mercator_device *>(object);
    if (id >= msc_object->object_values.size())
    {
        return -1;
    }

    Eigen::MatrixXd &M = msc_object->object_values[id];

    // save M to file in csv
    std::ofstream ofs(filename);
    ofs << M << std::endl;
    ofs.close();
    return true;
}

static search_device *__msc_create();
static void __msc_destroy(search_device *object);

static search_module __msc_module = {
    .name = mercator_module_name,
    .create_object = &__msc_create_object,
    .search = &__msc_search,
    .config = &__msc_config,
    .join = &__msc_join,
    .join_flush = &__msc_join_flush,

    .serialize = &__msc_serialize,
    .deserialize = &__msc_deserialize,
    .save = &__msc_save,

    .create = &__msc_create,
    .destroy = &__msc_destroy,
};

static search_device *__msc_create()
{
    auto object = new mercator_device;
    object->module = &__msc_module;
    return object;
}

static void __msc_destroy(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, mercator_module_name) != 0)
    {
        return;
    }

    delete static_cast<mercator_device *>(object);
}

register_search_module(__msc_module);
