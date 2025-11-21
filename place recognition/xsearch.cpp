#include <xsearch.h>

static const search_module *search_modules[20] = {nullptr};
static size_t search_modules_count = 0;

void __register_search_module(const search_module *module_) {
    search_modules[search_modules_count++] = module_;
}

search_device* create_search_device(const char *name) {
    for (size_t i = 0; i < search_modules_count; ++i) {
        if (strcmp(search_modules[i]->name, name) == 0) {
            return search_modules[i]->create();
        }
    }
    return nullptr;
}

void se_set_config(search_device *object, const xconfig& config) {
    for (auto& [k, v] : config) {
        if(strncmp(k.c_str(), "search.", 7) == 0) {
            if(object->module->config(object, k.c_str() + 7, v.c_str()))
                printf("\033[32maccepted config: %s = %s\033[0m\r\n", k.c_str() + 7, v.c_str());
            else
                printf("\033[31mignored config: %s = %s\033[0m\r\n", k.c_str() + 7, v.c_str());
        }
    }
}
