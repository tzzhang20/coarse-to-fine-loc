#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

#include <stdint.h>
#include <vector>

struct descriptor_header 
{
    uint32_t magic;
    uint32_t generator;
    uint16_t width, height;
    uint16_t count, length;
    char storage_path_suffix[128];
};

struct descriptor {
    char name[64];
    double timestamp;
    double x,y,z;
    double roll,pitch,yaw;
    float data[1200];
};

bool load_descriptor(const char* filename, std::vector<descriptor>& descriptors, descriptor_header& header);
bool save_descriptor(const char* filename, const char* suffix, const std::vector<descriptor>& descriptors);

#endif