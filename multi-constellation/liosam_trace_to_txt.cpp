#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;      

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,           // here we assume a XYZ + "test" (as fields)
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, roll, roll)
                                  (float, pitch, pitch)
                                  (float, yaw, yaw)
                                  (double, time, time)
);

#include <stdio.h>

struct xyzw {
    double x, y, z, w;
};

//Converts Roll, Pitch, Yaw (RPY) Euler angles to a quaternion
xyzw rpy_to_quat(double roll, double pitch, double yaw)
{
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    xyzw q;
    q.w = cy * cp * cr + sy * sp * sr;
    q.x = cy * cp * sr - sy * sp * cr;
    q.y = sy * cp * sr + cy * sp * cr;
    q.z = sy * cp * cr - cy * sp * sr;
    return q;
}

void print_traces(FILE* fp, const pcl::PointCloud<PointXYZIRPYT>& trace)
{
    
    for (auto& p : trace)
    {
        xyzw r = rpy_to_quat(p.roll, p.pitch, p.yaw);
        fprintf(fp, "%lf %f %f %f %f %f %f %f\n", p.time, p.x, p.y, p.z, r.x, r.y, r.z, r.w);
    }
}

//Save the LIOSAM mapping trajectory as a TXT file
int main(int argc, const char* const * argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    FILE* fp = fopen(argv[2], "w");
    if (!fp)
    {
        printf("Failed to open %s(%s)\n", argv[2], strerror(errno));
        return 1;
    }

    pcl::PointCloud<PointXYZIRPYT> trace;
    if (pcl::io::loadPCDFile(argv[1], trace) < 0)
    {
        printf("Failed to load %s\n", argv[1]);
        return 1;
    }

    print_traces(fp, trace);
    fclose(fp);
    return 0;
}
