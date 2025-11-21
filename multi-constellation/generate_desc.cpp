#include <filesystem>
#include "scancontext/Scancontext.h"
#include <pcl/io/pcd_io.h>

#include "descriptor.h"

struct trace_item {
    double time;
    float x;
    float y;
    float z;
    float roll, pitch, yaw;
};

//Due to the mismatch in data rates between GPS and LiDAR, the GPS coordinates for the point cloud data are interpolated.
bool gpslerp(const std::vector<trace_item>& traces, double t, trace_item* result) {
    if(traces.size() == 0 || t < traces[0].time || t > traces.back().time) {
        return false;
    }

    auto it = std::lower_bound(traces.begin(), traces.end(), t, [](const trace_item& trace, double t) {
        return trace.time < t;
    });

    if(it == traces.begin()) {
        *result = *it;
        return false;
    }

    auto prev = it - 1;

    double t0 = prev->time;
    double t1 = it->time;

    double w0 = (t1 - t) / (t1 - t0);

    result->time = t;
    result->x = prev->x * w0 + it->x * (1.0 - w0);
    result->y = prev->y * w0 + it->y * (1.0 - w0);
    result->z = prev->z * w0 + it->z * (1.0 - w0);

    result->roll = prev->roll * w0 + it->roll * (1.0 - w0);
    result->pitch = prev->pitch * w0 + it->pitch * (1.0 - w0);
    result->yaw = prev->yaw * w0 + it->yaw * (1.0 - w0);

    return true;
}

#include "files.h"

//load the timestamp from the data
std::vector<double> read_timestamp(const char* file) {
    FILE* fp = fopen(file, "r");
    if(!fp) {
        printf("Failed to open %s(%s)\n", file, strerror(errno));
        return {};
    }
    int frame_id= 0;
    double timestamp;
    std::vector<double> result;
    while(fscanf(fp, "%d %lf", &frame_id, &timestamp) == 2) {

        result.push_back(timestamp / 1.e9);
    }

    fclose(fp);
    return result;
}

//load the trajectories (x,y,z,roll,yaw,pitch,and xyz's covariance) from the data
void load_trace9(const char* filename, std::vector<trace_item>& items) {
    double cov_x, cov_y, cov_z;
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to open %s(%s)\n", filename, strerror(errno));
        return;
    }

    char buf[1024];
    while (fgets(buf, sizeof(buf), fp)) {
        trace_item item;
        if (sscanf(buf, "%lf,%f,%f,%f,%f,%f,%f,%lf,%lf,%lf", &item.time, &item.x, &item.y, &item.z, &item.roll, &item.pitch, &item.yaw, &cov_x, &cov_y, &cov_z) != 10) {
            printf("Failed to parse %s\n", buf);
            continue;
        }
        items.push_back(item);
    }

    fclose(fp);
}

std::vector<trace_item> downsample_traces(std::vector<trace_item>& input, const std::vector<double>& timestamp){
    std::vector<trace_item> result;
    for(auto val : timestamp) {
        trace_item item;
        if(gpslerp(input, val, &item)) {
            result.push_back(item);
        }
    }

    return result;
}

//Assign the corresponding pose and timestamp to the descriptor generated for each keyframe in the map.
int main(int argc , const char* const * argv) {
    if(argc < 4) {
        printf("Usage: %s <input_dir> <timestamp> <trace9>\n", argv[0]);
        return 1;
    }

    std::vector<descriptor> descriptors;
    const char* dir = argv[1];

    auto timestamp = read_timestamp(argv[2]);

    std::vector<trace_item> trace;
    load_trace9(argv[3], trace);

    auto files = list_files(dir, ".pcd");

    size_t processed = 0;
    progress_bar bar(files.size());
    if(files.size() != timestamp.size()) {
        printf("Timestamp count mismatch\n");
        return 1;
    }

    auto dt = downsample_traces(trace, timestamp);
    if(dt.size() != timestamp.size()) {
        printf("Failed to downsample trace\n");
        return 1;
    }

    for(auto iter : files) {
        std::filesystem::path p(iter);
        if(std::filesystem::is_directory(p)) {
            continue;
        }

        if(p.extension() != ".pcd") {
            continue;
        }

        pcl::PointCloud<XYZIRT> cloud;
        
        if(pcl::io::loadPCDFile(iter, cloud) < 0) {
            printf("Failed to load %s\n", iter.c_str());
            continue;
        }

        descriptor desc;
        memset(&desc, 0, sizeof(desc));

        auto matrix = SCManager::makeScancontext(cloud);

        std::copy(matrix.data(), matrix.data() + matrix.size(), desc.data);
        strcpy(desc.name, p.stem().string().c_str());

        desc.x = dt[processed].x;
        desc.y = dt[processed].y;
        desc.z = dt[processed].z;
        desc.roll = dt[processed].roll;
        desc.pitch = dt[processed].pitch;
        desc.yaw = dt[processed].yaw;
        desc.timestamp = timestamp[processed];

        descriptors.push_back(desc);
        bar.print_progress(++processed);
    }
    bar.done();
    printf("\n");
    if(!save_descriptor("scancontext.desc", dir, descriptors)) {
        printf("Failed to save scancontext.desc\n");
        return 1;
    }
}