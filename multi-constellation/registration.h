#ifndef __REGISTRATION_H__
#define __REGISTRATION_H__

#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include "common.h"

using RegistrationFun = Eigen::Matrix4d(*)(struct RegistrationHandleStruct*, struct RegistrationHandleStruct*, Eigen::Matrix4d);
using DeleteFun = void(*)(struct RegistrationHandleStruct*);

enum class RegistrationMethod {
    ICP,
    NDT,
    LOAM,
    KISS_ICP,
    TEASERPP
};

struct RegistrationHandleStruct {
    RegistrationMethod method;
    RegistrationFun fun;
    DeleteFun del;
};


typedef struct RegistrationHandleStruct* RegistrationHandle;


RegistrationHandle createRegistrationHandle(RegistrationMethod method, const pcl::PointCloud<XYZIRT>& target_cloud);

Eigen::Matrix4d registerCloud(RegistrationHandle source, RegistrationHandle target, Eigen::Matrix4d initial_guess);

void destroyRegistrationHandle(RegistrationHandle handle);

#endif // __REGISTRATION_H__