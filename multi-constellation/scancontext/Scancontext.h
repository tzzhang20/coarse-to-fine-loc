#pragma once

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


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "../common.h"

using SCPointType = XYZIRT;
using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;
// using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;

class SCManager
{
public: 
    SCManager( ) = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.


    static Eigen::MatrixXd makeScancontext(const pcl::PointCloud<SCPointType> & _scan_down );
    static Eigen::MatrixXd makeRingkeyFromScancontext(const  Eigen::MatrixXd &_desc );
    static Eigen::MatrixXd makeSectorkeyFromScancontext(const Eigen::MatrixXd &_desc );

    static int fastAlignUsingVkey (const Eigen::MatrixXd & _vkey1, const Eigen::MatrixXd & _vkey2 ); 
    static double distDirectSC (const Eigen::MatrixXd &_sc1,const Eigen::MatrixXd &_sc2 ); // "d" (eq 5) in the original paper (IROS 18)
    static std::pair<double, int> distanceBtnScanContext (const Eigen::MatrixXd &_sc1, const Eigen::MatrixXd &_sc2 ); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSaveScancontextAndKeys(const pcl::PointCloud<SCPointType> & _scan_down );
    void makeAndSaveScancontextAndKeys(const Eigen::MatrixXd& sc );
    std::pair<int, float> detectLoopClosureID( const Eigen::MatrixXd& sc  ); // int: nearest node index, float: relative yaw  

public:
    // hyper parameters ()
    static constexpr double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    static constexpr int    PC_NUM_RING = 20; // 20 in the original paper (IROS 18)
    static constexpr int    PC_NUM_SECTOR = 60; // 60 in the original paper (IROS 18)
    static constexpr double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    static constexpr double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    static constexpr double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    static constexpr int    NUM_CANDIDATES_FROM_TREE = 10; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    static constexpr double SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    static constexpr double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    // const double SC_DIST_THRES = 0.5; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // data 
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

    bool polarcontext_tree_built_ = false;

}; // SCManager

// } // namespace SC2
