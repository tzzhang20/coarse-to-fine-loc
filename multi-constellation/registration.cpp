template <typename T>
static inline auto p2(T x) { return x * x; }
#include "loam.hpp"

#include "registration.h"
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/registration/impl/icp.hpp>
#include <pcl/registration/impl/ndt.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/filters/impl/voxel_grid_covariance.hpp>

struct LOAMRegistrationHandleStruct : public RegistrationHandleStruct {
    cloud_featured<XYZIRT> cloud;
};

static Eigen::Matrix4d loam_registration(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess) {
    auto& source_cloud = ((LOAMRegistrationHandleStruct*)source)->cloud;
    auto& target_cloud = ((LOAMRegistrationHandleStruct*)target)->cloud;


    auto tr = LM(source_cloud, target_cloud, from_eigen(initial_guess));
    return to_eigen(tr);
}

static void loam_delete(RegistrationHandle handle) {
    delete (LOAMRegistrationHandleStruct*)handle;
}

static RegistrationHandle create_loam_registration_handle(const pcl::PointCloud<XYZIRT>& target_cloud) {
    pcl::PointCloud<XYZIRT> copy_target_cloud;
    std::copy(target_cloud.begin(), target_cloud.end(), std::back_inserter(copy_target_cloud.points));
    auto handle = new LOAMRegistrationHandleStruct();
    handle->method = RegistrationMethod::LOAM;
    handle->fun = &loam_registration;
    handle->del = &loam_delete;
    handle->cloud = get_features(copy_target_cloud.points.data(), copy_target_cloud.points.data() + copy_target_cloud.points.size());
    return handle;
}

struct DSRegistrationHandleStruct : public RegistrationHandleStruct {
    pcl::PointCloud<XYZIRT>::Ptr downsampled_cloud;
};

static RegistrationHandle DSCreateHandle(const pcl::PointCloud<XYZIRT>& target_cloud) {
    auto handle = new DSRegistrationHandleStruct();

    pcl::VoxelGrid<XYZIRT> downsample;
    downsample.setLeafSize(0.1, 0.1, 0.1);

    handle->downsampled_cloud = pcl::PointCloud<XYZIRT>::Ptr(new pcl::PointCloud<XYZIRT>());
    downsample.setInputCloud(target_cloud.makeShared());
    downsample.filter(*handle->downsampled_cloud);

    return handle;
}

static Eigen::Matrix4d ICP(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess) {
    pcl::IterativeClosestPoint<XYZIRT, XYZIRT> icp;
    icp.setInputSource(((DSRegistrationHandleStruct*)source)->downsampled_cloud);
    icp.setInputTarget(((DSRegistrationHandleStruct*)target)->downsampled_cloud);

    icp.align(*((DSRegistrationHandleStruct*)source)->downsampled_cloud, initial_guess.cast<float>());
    return icp.getFinalTransformation().cast<double>();
}

static Eigen::Matrix4d NDT(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess) {
    pcl::NormalDistributionsTransform<XYZIRT, XYZIRT> ndt;
    ndt.setInputSource(((DSRegistrationHandleStruct*)source)->downsampled_cloud);
    ndt.setInputTarget(((DSRegistrationHandleStruct*)target)->downsampled_cloud);

    ndt.align(*((DSRegistrationHandleStruct*)source)->downsampled_cloud, initial_guess.cast<float>());
    return ndt.getFinalTransformation().cast<double>();
}

static void delete_ds(RegistrationHandle handle) {
    delete (DSRegistrationHandleStruct*)handle;
}

#include <dlfcn.h>

struct FloatArrayRegistrationHandleStruct : public RegistrationHandleStruct {
    size_t size;
    float data[3];
};

static void teaser_delete(RegistrationHandle handle) {
    free(handle);
}

static void* libteaser = nullptr;

using solver = void(*)(const float*, const float*, size_t, size_t, double*);
static solver teaser_solver = nullptr;

static void load_teaser() {
    if(libteaser == nullptr) {
        libteaser = dlopen("libteaser_registration.so", RTLD_LAZY);
    }

    if(teaser_solver == nullptr) {
        teaser_solver = (solver)dlsym(libteaser, "teaser_solve");
    }
}

static Eigen::Matrix4d teaser_registration(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess) {
    load_teaser();

    auto src = (FloatArrayRegistrationHandleStruct*)source;
    auto tgt = (FloatArrayRegistrationHandleStruct*)target;

    Eigen::Matrix4d guess = initial_guess;

    double result[16];

    teaser_solver(src->data, tgt->data, src->size, tgt->size, result);

    Eigen::Matrix4d tr;
    for(size_t i = 0; i < 16; ++i) {
        tr(i / 4, i % 4) = result[i];
    }

    return tr;
}

static RegistrationHandle FloatArrayCreateHandle(const pcl::PointCloud<XYZIRT>& target_cloud) {
    auto handle = (FloatArrayRegistrationHandleStruct*)malloc(sizeof(FloatArrayRegistrationHandleStruct) + target_cloud.points.size() * 3 * sizeof(float));
    handle->method = RegistrationMethod::TEASERPP;
    handle->fun = &teaser_registration;
    handle->del = &teaser_delete;
    handle->size = target_cloud.points.size();
    for(size_t i = 0; i < target_cloud.points.size(); ++i) {
        handle->data[i * 3 + 0] = target_cloud.points[i].x;
        handle->data[i * 3 + 1] = target_cloud.points[i].y;
        handle->data[i * 3 + 2] = target_cloud.points[i].z;
    }
    return handle;
}


RegistrationHandle createRegistrationHandle(RegistrationMethod method, const pcl::PointCloud<XYZIRT>& target_cloud) {
    switch(method) {
    case RegistrationMethod::ICP:
    case RegistrationMethod::NDT:
        {
            auto handle = DSCreateHandle(target_cloud);
            handle->method = method;
            handle->fun = method == RegistrationMethod::ICP ? &ICP : &NDT;
            handle->del = &delete_ds;
            return handle;
        }
    case RegistrationMethod::LOAM:
        return create_loam_registration_handle(target_cloud);
    case RegistrationMethod::TEASERPP:
        return FloatArrayCreateHandle(target_cloud);
    }
    return nullptr;
}

Eigen::Matrix4d registerCloud(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess) {
    if(source->fun != target->fun) {
        return Eigen::Matrix4d::Identity();
    }

    return source->fun(source, target, initial_guess);
}

void destroyRegistrationHandle(RegistrationHandle handle) {
    handle->del(handle);
};
