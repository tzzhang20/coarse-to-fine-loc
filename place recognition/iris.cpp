#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fftm/fftm.hpp>

#include <xsearch.h>
#include <utils/serializer.h>
#include "iris_base.h"
#include <thread>

cv::Mat1b LidarIris::GetIris(const pcl::PointCloud<point_type> &cloud)
{
    cv::Mat1b IrisMap = cv::Mat1b::zeros(80, 360);

    for (auto &p : cloud.points)
    {
        float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);
        float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;
        int Q_dis = std::min(std::max((int)floor(dis), 0), 79);
        int Q_arc = std::min(std::max((int)ceil(p.z + 5), 0), 7);
        int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);
        IrisMap.at<uint8_t>(Q_dis, Q_yaw) |= (1 << Q_arc);
    }

    return IrisMap;
}

float LidarIris::Compare(const LidarIris::FeatureDesc &img1, const LidarIris::FeatureDesc &img2, int *bias) const
{
    if (_matchNum == 2) // 正向反向都有
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        float dis1;
        int bias1;
        GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);

        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);

        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;
        GetHammingDistance(img1.T, img1.M, T2x, M2x, secondShift, dis2, bias2);

        if (dis1 < dis2)
        {
            if (bias)
                *bias = bias1;
            return dis1;
        }
        else
        {
            if (bias)
                *bias = (bias2 + 180) % 360;
            return dis2;
        }
    }
    if (_matchNum == 1) // 只有反向
    {
        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);

        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;
        GetHammingDistance(img1.T, img1.M, T2x, M2x, secondShift, dis2, bias2);
        if (bias)
            *bias = (bias2 + 180) % 360;
        return dis2;
    }
    if (_matchNum == 0)
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        float dis1;
        int bias1;
        GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);
        if (bias)
            *bias = bias1;
        return dis1;
    }

    return std::numeric_limits<float>::max();
}

std::vector<cv::Mat2f> LidarIris::LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf)
{
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat2f filtersum = cv::Mat2f::zeros(1, cols);
    std::vector<cv::Mat2f> EO(nscale);
    int ndata = cols;
    if (ndata % 2 == 1)
        ndata--;
    cv::Mat1f logGabor = cv::Mat1f::zeros(1, ndata);
    cv::Mat2f result = cv::Mat2f::zeros(rows, ndata);
    cv::Mat1f radius = cv::Mat1f::zeros(1, ndata / 2 + 1);
    radius.at<float>(0, 0) = 1;
    for (int i = 1; i < ndata / 2 + 1; i++)
    {
        radius.at<float>(0, i) = i / (float)ndata;
    }
    double wavelength = minWaveLength;
    for (int s = 0; s < nscale; s++)
    {
        double fo = 1.0 / wavelength;
        double rfo = fo / 0.5;
        //
        cv::Mat1f temp; //(radius.size());
        cv::log(radius / fo, temp);
        cv::pow(temp, 2, temp);
        cv::exp((-temp) / (2 * log(sigmaOnf) * log(sigmaOnf)), temp);
        temp.copyTo(logGabor.colRange(0, ndata / 2 + 1));
        //
        logGabor.at<float>(0, 0) = 0;
        cv::Mat2f filter;
        cv::Mat1f filterArr[2] = {logGabor, cv::Mat1f::zeros(logGabor.size())};
        cv::merge(filterArr, 2, filter);
        filtersum = filtersum + filter;
        for (int r = 0; r < rows; r++)
        {
            cv::Mat2f src2f;
            cv::Mat1f srcArr[2] = {src.row(r).clone(), cv::Mat1f::zeros(1, src.cols)};
            cv::merge(srcArr, 2, src2f);
            cv::dft(src2f, src2f);
            cv::mulSpectrums(src2f, filter, src2f, 0);
            cv::idft(src2f, src2f);
            src2f.copyTo(result.row(r));
        }
        EO[s] = result.clone();
        wavelength *= mult;
    }
    filtersum = circShift(filtersum, 0, cols / 2);
    return EO;
}

void LidarIris::LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M)
{
    cv::Mat1f srcFloat;
    src.convertTo(srcFloat, CV_32FC1);
    auto list = LogGaborFilter(srcFloat, nscale, minWaveLength, mult, sigmaOnf);
    std::vector<cv::Mat1b> Tlist(nscale * 2), Mlist(nscale * 2);
    for (int i = 0; i < list.size(); i++)
    {
        cv::Mat1f arr[2];
        cv::split(list[i], arr);
        Tlist[i] = arr[0] > 0;
        Tlist[i + nscale] = arr[1] > 0;
        cv::Mat1f m;
        cv::magnitude(arr[0], arr[1], m);
        Mlist[i] = m < 0.0001;
        Mlist[i + nscale] = m < 0.0001;
    }
    cv::vconcat(Tlist, T);
    cv::vconcat(Mlist, M);
}

LidarIris::FeatureDesc LidarIris::GetFeature(const cv::Mat1b &src)
{
    FeatureDesc desc;
    desc.img = src;
    LoGFeatureEncode(src, _nscale, _minWaveLength, _mult, _sigmaOnf, desc.T, desc.M);
    return desc;
}

void LidarIris::GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias) const
{
    dis = NAN;
    bias = -1;
    for (int shift = scale - 2; shift <= scale + 2; shift++)
    {
        cv::Mat1b T1s = circShift(T1, 0, shift);
        cv::Mat1b M1s = circShift(M1, 0, shift);
        cv::Mat1b mask = M1s | M2;
        int MaskBitsNum = cv::sum(mask / 255)[0];
        int totalBits = T1s.rows * T1s.cols - MaskBitsNum;
        cv::Mat1b C = T1s ^ T2;
        C = C & ~mask;
        int bitsDiff = cv::sum(C / 255)[0];
        if (totalBits == 0)
        {
            dis = NAN;
        }
        else
        {
            float currentDis = bitsDiff / (float)totalBits;
            if (currentDis < dis || isnan(dis))
            {
                dis = currentDis;
                bias = shift;
            }
        }
    }
    return;
}

inline cv::Mat LidarIris::circRowShift(const cv::Mat &src, int shift_m_rows)
{
    if (shift_m_rows % src.rows == 0)
        return src.clone();
    shift_m_rows %= src.rows;
    int m = shift_m_rows > 0 ? shift_m_rows : src.rows + shift_m_rows;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range(src.rows - m, src.rows), cv::Range::all()).copyTo(dst(cv::Range(0, m), cv::Range::all()));
    src(cv::Range(0, src.rows - m), cv::Range::all()).copyTo(dst(cv::Range(m, src.rows), cv::Range::all()));
    return dst;
}

inline cv::Mat LidarIris::circColShift(const cv::Mat &src, int shift_n_cols)
{
    if (shift_n_cols % src.cols == 0)
        return src.clone();
    shift_n_cols %= src.cols;
    int n = shift_n_cols > 0 ? shift_n_cols : src.cols + shift_n_cols;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range::all(), cv::Range(src.cols - n, src.cols)).copyTo(dst(cv::Range::all(), cv::Range(0, n)));
    src(cv::Range::all(), cv::Range(0, src.cols - n)).copyTo(dst(cv::Range::all(), cv::Range(n, src.cols)));
    return dst;
}

cv::Mat LidarIris::circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols)
{
    return circColShift(circRowShift(src, shift_m_rows), shift_n_cols);
}

constexpr const char *iris_module_name = "iris";

struct iris_device : search_device
{
    LidarIris iris;
    std::vector<LidarIris::FeatureDesc> object_values;
    std::vector<object_id> joined_id;

    int parallel_threads = 1;

    iris_device() : iris(4, 18, 1.6, 0.75, 0)
    {
    }
};

static object_id __iris_create_object(search_device *object, const pcl::PointCloud<point_type> &cloud)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return obj_none;
    }

    auto iris_object = static_cast<iris_device *>(object);
    auto matrix = iris_object->iris.GetIris(cloud);
    auto feature = iris_object->iris.GetFeature(matrix);
    iris_object->object_values.push_back(std::move(feature));
    return iris_object->object_values.size() - 1;
}

static size_t __iris_search_1(iris_device *iris_object, object_id searched_target, search_result *results, size_t max_results)
{
    for (auto id : iris_object->joined_id)
    {
        int bias;
        float distance = iris_object->iris.Compare(
            iris_object->object_values[searched_target],
            iris_object->object_values[id],
            &bias);

        if (results->id == obj_none || distance < results->score)
        {
            results->id = id;
            results->score = distance;
            results->yaw = bias / 180.0f * M_PI;
        }
    }
    return 1;
}

static size_t __iris_search_parallel(iris_device *iris_object, object_id searched_target, search_result *results, size_t max_results)
{
    std::vector<search_result> r(iris_object->joined_id.size());
    size_t count = r.size();

    if (max_results > count)
    {
        max_results = count;
    }

    if (max_results == 0)
    {
        return 0;
    }

#pragma omp parallel for num_threads(iris_object->parallel_threads)
    for (size_t i = 0; i < count; i++)
    {
        int bias;
        size_t id = iris_object->joined_id[i];

        float distance = iris_object->iris.Compare(
            iris_object->object_values[searched_target],
            iris_object->object_values[id],
            &bias);

        r[i].id = id;
        r[i].score = distance;
        r[i].yaw = bias;
    }

    auto compare = [](const search_result &a, const search_result &b)
    { return a.score < b.score; };
    std::nth_element(r.begin(), r.begin() + max_results, r.end(), compare);
    std::sort(r.begin(), r.begin() + max_results, compare);

    for (size_t i = 0; i < max_results; i++)
    {
        results[i] = r[i];
    }

    return max_results;
}

static size_t __iris_search(search_device *object, object_id searched_target, search_result *results, size_t max_results)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return 0;
    }

    auto iris_object = static_cast<iris_device *>(object);

    if (searched_target >= iris_object->object_values.size())
    {
        return 0;
    }

    if (iris_object->parallel_threads == 1)
        return __iris_search_1(iris_object, searched_target, results, max_results);
    return __iris_search_parallel(iris_object, searched_target, results, max_results);
}

static bool __iris_config(search_device *object, const char *key, const char *value)
{
    if (strcmp(key, "threads") == 0)
    {
        int threads = std::stoi(value);
        if (threads < 0)
        {
            threads = std::thread::hardware_concurrency() - 1;
            printf("\033[92mSet numthreads to %d since config value < 0\033[0m\r\n", threads);
        }
        ((iris_device *)object)->parallel_threads = threads;
        return true;
    }
    return false;
}

static void __iris_join(search_device *object, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return;
    }

    auto iris_object = static_cast<iris_device *>(object);
    iris_object->joined_id.push_back(id);
}

static void __iris_join_flush(search_device *object)
{
}

static ssize_t __iris_serialize_featrue(FILE *fp, const LidarIris::FeatureDesc &f)
{
    ssize_t s1 = serialize_opencv(fp, f.img);
    if (s1 < 0)
    {
        return -1;
    }

    ssize_t s2 = serialize_opencv(fp, f.M);
    if (s2 < 0)
    {
        return -1;
    }

    ssize_t s3 = serialize_opencv(fp, f.T);
    if (s3 < 0)
    {
        return -1;
    }

    return s1 + s2 + s3;
}

static ssize_t __iris_serialize(search_device *object, FILE *fp, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return 0;
    }

    auto iris_object = static_cast<iris_device *>(object);
    if (id >= iris_object->object_values.size())
    {
        return -1;
    }

    auto &value = iris_object->object_values[id];
    return __iris_serialize_featrue(fp, value);
}

static ssize_t __iris_deserialize_featrue(FILE *fp, LidarIris::FeatureDesc &f)
{
    ssize_t s1 = deserialize_opencv(fp, f.img);
    if (s1 < 0)
    {
        return -1;
    }

    ssize_t s2 = deserialize_opencv(fp, f.M);
    if (s2 < 0)
    {
        return -1;
    }

    ssize_t s3 = deserialize_opencv(fp, f.T);
    if (s3 < 0)
    {
        return -1;
    }

    return s1 + s2 + s3;
}

static ssize_t __iris_deserialize(search_device *object, FILE *fp, object_id &id)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return 0;
    }

    auto iris_object = static_cast<iris_device *>(object);
    if (id >= iris_object->object_values.size())
    {
        return -1;
    }

    LidarIris::FeatureDesc value;
    auto s = __iris_deserialize_featrue(fp, value);
    if (s < 0)
    {
        return -1;
    }

    iris_object->object_values.push_back(std::move(value));
    id = iris_object->object_values.size() - 1;
    return s;
}

static bool __iris_save(search_device *object, object_id id, const char *filename)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return 0;
    }

    auto iris_object = static_cast<iris_device *>(object);
    if (id >= iris_object->object_values.size())
    {
        return -1;
    }

    auto &value = iris_object->object_values[id];

    // save as csv
    FILE *fp = fopen(filename, "w");
    if (fp == nullptr)
    {
        return false;
    }

    for (int i = 0; i < value.img.rows; i++)
    {
        for (int j = 0; j < value.img.cols; j++)
        {
            if (j != value.img.cols - 1)
                fprintf(fp, "%f,", value.img.at<float>(i, j));
            else
                fprintf(fp, "%f", value.img.at<float>(i, j));
        }
        fprintf(fp, "\r\n");
    }

    fclose(fp);
    return true;
}

static search_device *__iris_create();
static void __iris_destroy(search_device *object);

static search_module iris_module = {
    .name = iris_module_name,
    .create_object = &__iris_create_object,
    .search = &__iris_search,
    .config = &__iris_config,
    .join = &__iris_join,
    .join_flush = &__iris_join_flush,

    .serialize = &__iris_serialize,
    .deserialize = &__iris_deserialize,
    .save = &__iris_save,

    .create = &__iris_create,
    .destroy = &__iris_destroy,
};

static search_device *__iris_create()
{
    auto object = new iris_device();
    object->module = &iris_module;
    return object;
}

static void __iris_destroy(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, iris_module_name) != 0)
    {
        return;
    }

    delete static_cast<iris_device *>(object);
}

register_search_module(iris_module);
