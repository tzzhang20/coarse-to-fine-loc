#include "descriptor.h"
#include <stdio.h>
#include <string.h>
const uint32_t DESCRIPTOR_MAGIC = 0x5343414E;

int load_header(FILE* fp, descriptor_header* header) {
    int result = fread(header, sizeof(descriptor_header), 1, fp);
    if (result != 1) {
        printf("Failed to read header\n");
        return -1;
    }

    if (header->magic != DESCRIPTOR_MAGIC) {
        printf("Invalid magic\n");
        return -1;
    }

    if(header->generator != 0x536D6D60) {
        printf("Invalid generator\n");
        return -1;
    }

    return 0;
}

//Read the descriptors generated during the mapping process.
bool load_descriptor(const char* filename, std::vector<descriptor>& descriptors, descriptor_header& header) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Failed to open %s\n", filename);
        return false;
    }

    if (load_header(fp, &header) < 0) {
        fclose(fp);
        return false;
    }

    descriptors.resize(header.count);
    
    int count = fread(descriptors.data(), sizeof(descriptor), header.count, fp);
    if (count != header.count) {
        printf("Failed to read descriptors\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}


//Save the descriptors generated during the mapping process.
bool save_descriptor(const char* filename, const char* suffix, const std::vector<descriptor>& descriptors) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open %s\n", filename);
        return false;
    }

    descriptor_header header;
    header.magic = DESCRIPTOR_MAGIC;
    header.generator = 0x536D6D60;
    header.width = 60;
    header.height = 20;
    header.count = descriptors.size();
    header.length = 1200;
    strncpy(header.storage_path_suffix, suffix, sizeof(header.storage_path_suffix));

    int result = fwrite(&header, sizeof(descriptor_header), 1, fp);
    if (result != 1) {
        printf("Failed to write header\n");
        fclose(fp);
        return false;
    }

    result = fwrite(descriptors.data(), sizeof(descriptor), descriptors.size(), fp);
    if (result != descriptors.size()) {
        printf("Failed to write descriptors\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

