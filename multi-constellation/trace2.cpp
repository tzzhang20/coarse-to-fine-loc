#include <stdio.h>
#include <vector>
#include <string.h>
#include <errno.h>
#include <eigen3/Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include "view_interface"
#include "descriptor.h"
#include "circle3.hpp"
#include <filesystem>
#include "scancontext/Scancontext.h"
#include "loam.hpp"
#include "registration.h"
#include "datasource.h"

struct cloud_cache
{
    int age = 0;
    RegistrationHandle cloud;
    std::string name;
};

struct cloud_cache_table
{
    DataSource source;
    RegistrationMethod method;
    cloud_cache items[20];
    size_t count = 0;

    cloud_cache *begin()
    {
        return items;
    }

    cloud_cache *end()
    {
        return items + count;
    }
};

void incr_age(cloud_cache_table &table)
{
    for (auto &cache : table)
    {
        cache.age++;
    }
}

void load_cloud(cloud_cache_table &table, const char *path, RegistrationHandle *pcloud)
{
    auto it = std::find_if(std::begin(table), std::end(table), [path](const cloud_cache &cache)
                           { return cache.name == path; });

    if (it != std::end(table))
    {
        pcloud[0] = it->cloud;
        incr_age(table);
        it->age = 0;
        return;
    }
    pcl::PointCloud<XYZIRT> cloud;
    cloud_cache *replace = nullptr;
    if (table.count == 20)
    {
        auto it = std::max_element(table.begin(), table.end(), [](const cloud_cache &a, const cloud_cache &b)
                                   { return a.age < b.age; });

        replace = it;
    }
    else
    {
        replace = table.items + table.count;
        replace->cloud = nullptr;
        table.count++;
    }

    incr_age(table);

    replace->age = 0;
    if (table.source->reader(path, cloud) < 0)
    {
        printf("Failed to load %s\n", path);
        return;
    }
    if (replace->cloud != nullptr)
        destroyRegistrationHandle(replace->cloud);
    replace->cloud = createRegistrationHandle(table.method, cloud);
    replace->name = path;

    pcloud[0] = replace->cloud;
}

struct circle3_param
{
    RegistrationHandle cloud;
    double x, y, z;
};

inline float calculate_R(RegistrationHandle source_feature, RegistrationHandle param, float init_yaw)
{
    Transform init_transform;
    memset(&init_transform, 0, sizeof(init_transform));
    init_transform.yaw = init_yaw;

    Eigen::Matrix4d transform = registerCloud(source_feature, param, to_eigen(init_transform));
    float x = transform(0, 3);
    float y = transform(1, 3);
    float z = transform(2, 3);

    return std::sqrt(x * x + y * y + z * z);
}

Point3 call_circle3(RegistrationHandle source_feature, const std::vector<circle3_param> &params, float init_yaw, bool print_loss = false)
{
    std::vector<Circle3> observed_circles;
    for (auto &param : params)
    {
        Circle3 observed_circle;
        observed_circle.x = param.x;
        observed_circle.y = param.y;
        observed_circle.z = param.z;
        observed_circle.radius = calculate_R(source_feature, param.cloud, init_yaw);
        observed_circles.push_back(observed_circle);
        if (print_loss)
        {
            printf("circle: %f %f %f %f\n", observed_circle.x, observed_circle.y, observed_circle.z, observed_circle.radius);
        }
    }

    auto [res, ok] = solve_circles(observed_circles.data(), observed_circles.size());
    if (ok)
    {
        return res;
    }

    return {0.0f, 0.0f, 0.0f};
}

const char *last_slash(const char *path);

struct TestSet
{
    std::vector<std::string> paths;
    bool contains(const char *path)
    {
        return std::find(paths.begin(), paths.end(), last_slash(path)) != paths.end();
    }
};

void remove_blank(char *str)
{
    for (char *p = str + strlen(str) - 1; p >= str; p--)
    {
        if (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t')
        {
            *p = '\0';
        }
        else
        {
            break;
        }
    }
}

TestSet loadTestSet(const char *file, DataSourceType type)
{
    struct TestSet set;
    FILE *fp = fopen(file, "r");

    if (!fp)
    {
        printf("Failed to open %s: %s\n", file, strerror(errno));
        return set;
    }

    // This code creates a new array of 1024 bytes, and then
    // reads data from the file into the array.  The data is
    // then processed in some way, and then written to the
    // output file.

    char buf[1024];
    int num = 0;
    while (fgets(buf, sizeof(buf), fp))
    {
        remove_blank(buf);
        set.paths.push_back(last_slash(buf));
    }
    fclose(fp);
    return set;
}

DataSourceType get_type(const char *name)
{
    if (strcmp(name, "bin") == 0)
    {
        return DataSourceType::BIN_Directory;
    }
    else if (strcmp(name, "pcd") == 0)
    {
        return DataSourceType::PCD_Directory;
    }
    else
    {
        printf("Unknown type %s\n", name);
        exit(1);
    }
    return DataSourceType::BIN_Directory;
}

int main(int argc, const char *const *argv)
{
    if (argc < 7)
    {
        printf("Usage: %s <input_dir> <timestamp> <trace9> <output> <type> <testset>\n", argv[0]);
        return 1;
    }

    DataSource source = createDataSource(get_type(argv[5]), argv[1], argv[2], argv[3]);

    if (!source)
    {
        printf("Failed to create data source\n");
        return 1;
    }

    TestSet test_set = loadTestSet(argv[6], get_type(argv[5]));
    if (test_set.paths.empty())
    {
        printf("Failed to load test set\n");
        return 1;
    }

    SCManager sc;
    std::vector<size_t> inner_clouds, outer_clouds;
    std::vector<size_t> object_flag;
    pcl::PointCloud<XYZIRT> cloud;

    progress_bar sc_bar(source->paths.size());
    for (size_t i = 0; i < source->paths.size(); i++)
    {
        if (i % 2 == 0 || test_set.contains(source->paths[i].c_str()))
        {
            object_flag.push_back(0);
            outer_clouds.push_back(i);
            continue;
        }
        object_flag.push_back(1);
        inner_clouds.push_back(i);
        if (source->reader(source->paths[i].c_str(), cloud) < 0)
        {
            printf("Failed to load %s\n", source->paths[i].c_str());
            return 1;
        }
        sc.makeAndSaveScancontextAndKeys(cloud);
        sc_bar.print_progress(i);
    }
    sc_bar.done();

    cloud_cache_table table;
    table.source = source;
    table.method = RegistrationMethod::LOAM;

    printf("Processing %lu files\n", object_flag.size());
    progress_bar circle3_progess(object_flag.size());
    FILE *fp = fopen(argv[4], "w");

    auto now = std::chrono::system_clock::now();
    for (size_t i = 0; i < object_flag.size(); i++)
    {

        if (object_flag[i] == 1)
        {
            continue;
        }
        if (!test_set.contains(source->paths[i].c_str()))
            continue;

        pcl::PointCloud<XYZIRT> cloud;
        if (!source->reader(source->paths[i].c_str(), cloud))
        {
            printf("Failed to load %s\n", source->paths[i].c_str());
            continue;
        }

        Eigen::MatrixXd m = SCManager::makeScancontext(cloud);
        auto result = sc.detectLoopClosureID(m);

        if (result.first == -1)
        {
            fprintf(fp, "%s,-1,0,0,0,0,0,0,0\n", source->paths[i].c_str());
        }
        else
        {
            constexpr int circle3_count = 8;

            int start = result.first > circle3_count / 2 ? result.first - circle3_count / 2 : 0;
            int end = start + circle3_count;

            if (end > inner_clouds.size())
            {
                end = inner_clouds.size();
                start = end - circle3_count;
            }

            std::vector<circle3_param> params;

            for (int i = start; i < end; i++)
            {
                size_t inner_index = inner_clouds[i];
                circle3_param param;
                load_cloud(table, source->paths[i].c_str(), &param.cloud);
                param.x = source->traces[inner_index](0, 3);
                param.y = source->traces[inner_index](1, 3);
                param.z = source->traces[inner_index](2, 3);
                params.push_back(param);
            }

            auto handle = createRegistrationHandle(table.method, cloud);
            auto p = call_circle3(handle, params, result.second / 180.0 * M_PI, i == 2403);
            double gt_x = source->traces[i](0, 3);
            double gt_y = source->traces[i](1, 3);
            double gt_z = source->traces[i](2, 3);

            double loss = std::sqrt((p.x - gt_x) * (p.x - gt_x) + (p.y - gt_y) * (p.y - gt_y) + (p.z - gt_z) * (p.z - gt_z));
            if (i == 2403)
            {
                printf("p: %lf %lf %lf\n", p.x, p.y, p.z);
                printf("gt: %lf %lf %lf\n", gt_x, gt_y, gt_z);
                printf("loss: %lf\n", loss);
                printf("yaw: %lf\n", result.second / 180.0 * M_PI);
            }
            fprintf(fp, "%s,%zd,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", source->paths[i].c_str(), inner_clouds[result.first], p.x, p.y, p.z, gt_x, gt_y, gt_z, loss);
            destroyRegistrationHandle(handle);
        }

        circle3_progess.print_progress(i);
        fflush(fp);
    }

    auto end = std::chrono::system_clock::now();
    double duration = (end - now).count() / (double)std::chrono::system_clock::period::den;

    fprintf(fp, "Total time: %lf\n", duration);
    fclose(fp);
    circle3_progess.done();
    printf("\n");
    return 0;
}
