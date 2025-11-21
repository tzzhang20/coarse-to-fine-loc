#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <nanoflann.hpp>
#include <fftm/fftm.hpp>
#include <algorithm>
#include <xsearch.h>
#include <nanoflann.hpp>
#include <utils/KDTreeVectorOfVectorsAdaptor.h>
#include <utils/serializer.h>

using namespace std;
using namespace cv;

struct FeatureDesc
{
    cv::Mat1b img;
    cv::Mat1b T;
    cv::Mat1b M;
};

class MDFSC
{
public:
    MDFSC(int nscale, int minWaveLength, float mult, float sigmaOnf, int matchNum) : _nscale(nscale),
                                                                                    _minWaveLength(minWaveLength),
                                                                                    _mult(mult),
                                                                                    _sigmaOnf(sigmaOnf),
                                                                                    _matchNum(matchNum)
    {
    }
    MDFSC(const MDFSC &) = delete;
    MDFSC &operator=(const MDFSC &) = delete;

    static cv::Mat1b GetIris(const pcl::PointCloud<point_type> &cloud);
    float Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias = nullptr);

    FeatureDesc GetFeature(const cv::Mat1b &src);
    std::vector<cv::Mat2f> LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf);
    void GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias);

    static inline cv::Mat circRowShift(const cv::Mat &src, int shift_m_rows);
    static inline cv::Mat circColShift(const cv::Mat &src, int shift_n_cols);
    static cv::Mat circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols);

private:
    void LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M);

    int _nscale;
    int _minWaveLength;
    float _mult;
    float _sigmaOnf;
    int _matchNum;
};

//Initialize the descriptor as an 80x360 matrixas, as presented in Equations 4 and 5.
cv::Mat1b MDFSC::GetIris(const pcl::PointCloud<point_type> &cloud)
{
    cv::Mat1b IrisMap = cv::Mat1b::zeros(80, 360);

    for (auto p : cloud.points)
    {
        float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);
        float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;
        int Q_dis = std::min(std::max((int)floor(dis), 0), 79);
        int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);
        uint8_t thisz = IrisMap.at<uint8_t>(Q_dis, Q_yaw);
	
       //Non-uniform partitioning
        if (p.data[2] + 2 > thisz)
        {
            if (Q_dis <= 19)
            {
                int c = Q_yaw / 8 * 8;
                for (int i = 0; i < 8; i++)
                    IrisMap.at<uint8_t>(Q_dis, c + i) = p.data[2] + 2;
            }
            else
            {
                IrisMap.at<uint8_t>(Q_dis, Q_yaw) = p.data[2] + 2;
            }
        }
    }

    return IrisMap;
}


//Compute the similarity between descriptors, as presented in Equations 10-16.
float MDFSC::Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias)
{
    if (_matchNum == 2)  //For same-direction loop closure
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
    if (_matchNum == 1) //For opposite-direction loop closure
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
    if (_matchNum == 0) //others
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        // std::cout<<firstShift<<std::endl;
        float dis1;
        int bias1;
        GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);
        if (bias)
            *bias = bias1;
        return dis1;
    }

    return 0.0f;
}

//Extract log-Gabor features
std::vector<cv::Mat2f> MDFSC::LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf)
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

//The log-Gabor feature encoding for similarity calculation follows Equation 16.
void MDFSC::LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M)
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

FeatureDesc MDFSC::GetFeature(const cv::Mat1b &src)
{
    FeatureDesc desc;
    desc.img = src;
    LoGFeatureEncode(src, _nscale, _minWaveLength, _mult, _sigmaOnf, desc.T, desc.M);
    return desc;
}

//The Hamming distance calculation for Equation 16.
void MDFSC::GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias)
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

inline cv::Mat MDFSC::circRowShift(const cv::Mat &src, int shift_m_rows)
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

inline cv::Mat MDFSC::circColShift(const cv::Mat &src, int shift_n_cols)
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

cv::Mat MDFSC::circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols)
{
    return circColShift(circRowShift(src, shift_m_rows), shift_n_cols);
}

using vtype = std::array<float, 80>;

struct __mdfsc_device : search_device
{
    KDTreeVectorOfVectorsAdaptor<std::vector<vtype>, float, 80, nanoflann::metric_L2> *tree = nullptr;
    MDFSC mdfsc;
    std::vector<FeatureDesc> features;
    std::vector<vtype> joined_vtype;
    std::vector<object_id> joined_id;

    __mdfsc_device() : mdfsc(4, 18, 1.6, 0.75, 0) {}
};

constexpr const char *__mdfsc_module_name = "mdfsc";

static vtype __mdfsc_makev(const FeatureDesc &desc)
{
    vtype v;
    assert(desc.img.rows == 80);

    for (int i = 0; i < 80; i++)
    {
        v[i] = 0;
        for (int j = 0; j < desc.img.cols; j++)
        {
            v[i] += (float)desc.img.at<uchar>(i, j);
        }
    }
    return v;
}

static object_id __mdfsc_create_object(search_device *object, const pcl::PointCloud<point_type> &cloud)
{
    if (object->module->name != __mdfsc_module_name)
    {
        return obj_none;
    }

    auto mdfsc = static_cast<__mdfsc_device *>(object);
    auto key = mdfsc->mdfsc.GetIris(cloud);
    auto desc = mdfsc->mdfsc.GetFeature(key);

    mdfsc->features.push_back(std::move(desc));
    return mdfsc->features.size() - 1;
}

//Descriptor retrieval, as presented in Equations 8-16.
static size_t __mdfsc_search(search_device *object, object_id searched_target, search_result *results, size_t max_results)
{
    if (object->module->name != __mdfsc_module_name)
    {
        return 0;
    }

    auto mdfsc = static_cast<__mdfsc_device *>(object);
    auto &target = mdfsc->features[searched_target];

    if (mdfsc->joined_id.size() < max_results)
    {
        max_results = mdfsc->joined_id.size();
    }

    if (max_results == 0 || mdfsc->tree == nullptr)
    {
        return 0;
    }

    size_t candidate_count = max_results < 5 ? 10 : max_results * 2;

    std::vector<std::pair<size_t, double>> ret_matches;
    vtype v = __mdfsc_makev(target);
    std::vector<size_t> index(candidate_count);
    std::vector<float> dist(candidate_count);

    candidate_count = mdfsc->tree->index->knnSearch(v.data(), candidate_count, index.data(), dist.data(), 10);

    std::vector<search_result> result_list(candidate_count);
    for (size_t i = 0; i < candidate_count; i++)
    {
        object_id cid = mdfsc->joined_id[index[i]];
        int bias = 0;
        float distance = mdfsc->mdfsc.Compare(target, mdfsc->features[cid], &bias);

        result_list[i].id = cid;
        result_list[i].score = distance;
        result_list[i].yaw = bias / 180.0f * M_PI;
    }

    std::sort(result_list.begin(), result_list.end(), [](const search_result &a, const search_result &b)
              { return a.score < b.score; });

    for (size_t i = 0; i < max_results; i++)
    {
        results[i] = result_list[i];
    }

    return max_results;
}

static bool __mdfsc_config(search_device *object, const char *key, const char *value)
{
    return false;
}

static void __mdfsc_join(search_device *object, object_id id)
{
    if (object->module->name != __mdfsc_module_name)
    {
        return;
    }

    auto mdfsc = static_cast<__mdfsc_device *>(object);
    mdfsc->joined_id.push_back(id);
    mdfsc->joined_vtype.push_back(__mdfsc_makev(mdfsc->features[id]));
}

static void __mdfsc_join_flush(search_device *object)
{
    if (object->module->name != __mdfsc_module_name)
    {
        return;
    }

    auto mdfsc = static_cast<__mdfsc_device *>(object);

    if (mdfsc->tree != nullptr)
    {
        delete mdfsc->tree;
    }
    mdfsc->tree = new KDTreeVectorOfVectorsAdaptor<std::vector<vtype>, float, 80, nanoflann::metric_L2>(80, mdfsc->joined_vtype, 10);
    mdfsc->tree->index->buildIndex();
}

static ssize_t __mdfsc_serialize_featrue(FILE *fp, const FeatureDesc &f)
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

static ssize_t __mdfsc_serialize(search_device *object, FILE *fp, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, __mdfsc_module_name) != 0)
    {
        return 0;
    }

    auto mdfsc_object = static_cast<__mdfsc_device *>(object);
    if (id >= mdfsc_object->features.size())
    {
        return -1;
    }

    auto &value = mdfsc_object->features[id];
    return __mdfsc_serialize_featrue(fp, value);
}

static ssize_t __mdfsc_deserialize_featrue(FILE *fp, FeatureDesc &f)
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

static ssize_t __mdfsc_deserialize(search_device *object, FILE *fp, object_id &id)
{
    if (object == nullptr || strcmp(object->module->name, __mdfsc_module_name) != 0)
    {
        return 0;
    }

    auto mdfsc_object = static_cast<__mdfsc_device *>(object);
    if (id >= mdfsc_object->features.size())
    {
        return -1;
    }

    FeatureDesc value;
    auto s = __mdfsc_deserialize_featrue(fp, value);
    if (s < 0)
    {
        return -1;
    }

    mdfsc_object->features.push_back(std::move(value));
    id = mdfsc_object->features.size() - 1;
    return s;
}

static search_device *__mdfsc_create();

static void __mdfsc_destroy(search_device *object)
{
    if (object->module->name != __mdfsc_module_name)
    {
        return;
    }

    auto mdfsc = static_cast<__mdfsc_device *>(object);
    if (mdfsc->tree != nullptr)
    {
        delete mdfsc->tree;
    }
    delete mdfsc;
}

static search_module __mdfsc_module = {
    .name = __mdfsc_module_name,

    .create_object = __mdfsc_create_object,
    .search = __mdfsc_search,
    .config = __mdfsc_config,
    .join = __mdfsc_join,
    .join_flush = __mdfsc_join_flush,

    .serialize = __mdfsc_serialize,
    .deserialize = __mdfsc_deserialize,
    .save = nullptr,

    .create = __mdfsc_create,
    .destroy = __mdfsc_destroy};

static search_device *__mdfsc_create()
{
    auto mdfsc = new __mdfsc_device();
    mdfsc->module = &__mdfsc_module;
    return mdfsc;
}

register_search_module(__mdfsc_module);
