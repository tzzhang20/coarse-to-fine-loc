#include <Eigen/Dense>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <utils/KDTreeVectorOfVectorsAdaptor.h>

struct Ranged
{
    float x;
    float y;
    float z;
    float intensity;
    std::uint16_t row;
    std::uint16_t col;
    std::uint32_t t;
    std::uint16_t label;
};

struct fresco_config
{
    int GROUND_HEIGHT_GRID_ROWS = 75;
    int GROUND_HEIGHT_GRID_COLS = 50;
    int Horizon_SCAN = 2083; // for hdl 64e
    int N_SCAN = 64;
};

static cv::Mat bev_img(const std::vector<Ranged> &cloud, float interval = 2.0f)
{
    static int MAX_RANGE = 100;
    static int MAT_SIZE = MAX_RANGE * 2 / interval + 1;
    cv::Mat cart_bv = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32F);

    for (auto &pi : cloud)
    {
        int x = round((pi.x + MAX_RANGE) / interval + 0.5);
        int y = round((pi.y + MAX_RANGE) / interval + 0.5);

        if (x < 0 || x >= MAT_SIZE || y < 0 || y >= MAT_SIZE || pi.label == 0)
        {
            continue;
        }

        if (pi.z + 2.0f > cart_bv.at<float>(x, y))
        {
            cart_bv.at<float>(x, y) = pi.z + 2.0f;
        }
    }

    return cart_bv;
}

static void order_cloud(const fresco_config &config,
                        const std::vector<Ranged> &input_cloud,
                        std::vector<Ranged> &output_cloud)
{
    output_cloud.resize(config.N_SCAN * config.Horizon_SCAN);

    // zeros
    for (auto &point : output_cloud)
    {
        point.x = 0.0f;
        point.y = 0.0f;
        point.z = 0.0f;
        point.intensity = 0.0f;
        point.row = 0;
        point.col = 0;
        point.t = 0;
        point.label = 0;
    }

    // int numberOfCores = 8;
    // #pragma omp parallel for num_threads(numberOfCores)
    for (auto &point : input_cloud)
    {
        int row_idx = point.row;
        int col_idx = point.col;

        int point_idx = row_idx * config.Horizon_SCAN + col_idx;

        output_cloud[point_idx] = point;
    }
}

static inline std::pair<int, int> getBelongingGrid(const Ranged &r)
{
    int sector_row_idx = 0;
    int sector_col_idx = 0;

    float normalized_x = r.x + 75.0;
    float normalized_y = r.y + 50.0;

    sector_row_idx = static_cast<int>(std::floor(normalized_x / 2.0));
    sector_col_idx = static_cast<int>(std::floor(normalized_y / 2.0));

    if (sector_row_idx >= 75)
    {
        sector_row_idx = 75 - 1;
    }
    if (sector_row_idx < 0)
    {
        sector_row_idx = 0;
    }

    if (sector_col_idx >= 50)
    {
        sector_col_idx = 50 - 1;
    }
    if (sector_col_idx < 0)
    {
        sector_col_idx = 0;
    }

    return std::make_pair(sector_row_idx, sector_col_idx);
}

static void mark_ground(
    const fresco_config &config,
    std::vector<Ranged> &output_cloud)
{
    Eigen::MatrixXi ground_mat(config.N_SCAN, config.Horizon_SCAN);
    ground_mat.setZero();

    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    // std::sort(output_cloud->points.begin(), output_cloud->points.end(), [](pcl::PointXYZIRCT &p1, pcl::PointXYZIRCT &p2) -> bool {
    //     return (p1.row < p2.row) || (p1.row == p2.row && p1.col < p2.col);
    // });

    // used to compute average ground height
    Eigen::MatrixXf ground_grid_avg_heights(config.GROUND_HEIGHT_GRID_ROWS, config.GROUND_HEIGHT_GRID_COLS);
    ground_grid_avg_heights.setZero();
    /*
    cv::Mat num_ground_grid_points = 0.01 * cv::Mat::ones(
                                                GROUND_HEIGHT_GRID_ROWS, GROUND_HEIGHT_GRID_COLS, CV_32F);
    */

    Eigen::MatrixXi num_ground_grid_points(config.GROUND_HEIGHT_GRID_ROWS, config.GROUND_HEIGHT_GRID_COLS);
    num_ground_grid_points.setZero();

    // FIXME: should change groundScanInd for hdl-64e
    int groundScanInd = 50;
    for (int col_idx = 0; col_idx < config.Horizon_SCAN; col_idx++)
    {
        for (int row_idx = config.N_SCAN - 1; row_idx > config.N_SCAN - groundScanInd - 1; row_idx--)
        {

            lowerInd = row_idx * config.Horizon_SCAN + col_idx;
            upperInd = (row_idx - 1) * config.Horizon_SCAN + col_idx;

            // 防止正上方有一个地面点没有读数，使用相邻点替代
            if (output_cloud[upperInd].intensity == -1)
            {
                int tmp_col_idx = (col_idx + 2) % config.Horizon_SCAN;
                upperInd = (row_idx - 1) * config.Horizon_SCAN + tmp_col_idx;
            }

            if (output_cloud[upperInd].intensity == -1)
            {
                int tmp_col_idx = (col_idx - 2) % config.Horizon_SCAN;
                upperInd = (row_idx - 1) * config.Horizon_SCAN + tmp_col_idx;
            }

            // use point on the other ring
            if (output_cloud[upperInd].intensity == -1 && row_idx >= 2)
            {
                int tmp_row_idx = row_idx - 2;
                upperInd = tmp_row_idx * config.Horizon_SCAN + col_idx;
            }

            if (output_cloud[lowerInd].intensity == -1 ||
                output_cloud[upperInd].intensity == -1)
            {
                // no info to check, invalid points
                ground_mat(row_idx, col_idx) = -1;
                continue;
            }

            diffX = output_cloud[upperInd].x - output_cloud[lowerInd].x;
            diffY = output_cloud[upperInd].y - output_cloud[lowerInd].y;
            diffZ = output_cloud[upperInd].z - output_cloud[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180.0 / M_PI;

            float sensorMountAngle = 0.0f;
            // float angle_thres = 0.4 * range_mat_.at<float>(row_idx, col_idx) + 8.8;

            // mark as ground points
            if (abs(angle - sensorMountAngle) <= 10.0f)
            {
                ground_mat(row_idx, col_idx) = 1;
                ground_mat(row_idx - 1, col_idx) = 1;
            }
        }
    }

    // 分块求地面高度平均值
    for (int row_idx = 0; row_idx < config.N_SCAN; row_idx++)
    {
        for (int col_idx = 0; col_idx < config.Horizon_SCAN; col_idx++)
        {
            if (ground_mat(row_idx, col_idx) != 1)
            {
                continue;
            }
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * config.Horizon_SCAN + col_idx;
            std::tie(sector_row, sector_col) = getBelongingGrid(output_cloud[point_index]);
            ground_grid_avg_heights(sector_row, sector_col) +=
                output_cloud[point_index].z;
            // min height instead
            // if (output_cloud->points[point_index].z < ground_grid_avg_heights.at<float>(sector_row, sector_col)) {
            //     ground_grid_avg_heights.at<float>(sector_row, sector_col) = output_cloud->points[point_index].z;
            // }

            num_ground_grid_points(sector_row, sector_col) += 1;
        }
    }

    // ground_grid_avg_heights = ground_grid_avg_heights / num_ground_grid_points;
    for (int i = 0; i < ground_grid_avg_heights.rows(); i++)
    {
        for (int j = 0; j < ground_grid_avg_heights.cols(); j++)
        {
            if (num_ground_grid_points(i, j) != 0)
            {
                ground_grid_avg_heights(i, j) /= num_ground_grid_points(i, j);
            }
        }
    }

    // std::cout << "ground_sector_height: \n" << ground_grid_avg_heights_ << std::endl;

    // extract ground cloud (ground_mat_ == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~N_SCAN-1, need range_mat_ for mark label matrix for the 16th scan
    constexpr std::pair<int, int> four_neighbor_iterator_[] = {
        {-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    for (int row_idx = 0; row_idx < config.N_SCAN; row_idx++)
    {
        for (int col_idx = 0; col_idx < config.Horizon_SCAN; col_idx++)
        {

            // 防止车顶被当作地面
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * config.Horizon_SCAN + col_idx;
            std::tie(sector_row, sector_col) = getBelongingGrid(output_cloud[point_index]);

            int neighbor_sector_row = 0;
            int neighbor_sector_col = 0;
            for (auto iter : four_neighbor_iterator_)
            {
                neighbor_sector_row = sector_row + iter.first;
                neighbor_sector_col = sector_col + iter.second;

                if (neighbor_sector_row < 0 || neighbor_sector_row >= 75 ||
                    neighbor_sector_col < 0 || neighbor_sector_col >= 50)
                {
                    continue;
                }
                // 检查高度：是否比周围地面分块的高度高很多
                if (output_cloud[point_index].z -
                        ground_grid_avg_heights(neighbor_sector_row, neighbor_sector_col) >
                    0.30)
                {
                    ground_mat(row_idx, col_idx) = 0;
                    break;
                }
            }

            // ground points are labeled 0 in label_mat
            if (ground_mat(row_idx, col_idx) == 1)
            {
                output_cloud[point_index].label = 0; // 0 means gound points
            }
            else
            {
                // output_cloud->points[point_index].label = 1; // 1 means non-gound points
            }
        }
    }
}

static inline float makeAngleSemiPositive(float input_angle)
{
    if (input_angle >= 360.0f)
    {
        return (input_angle - 360.0f);
    }
    else if (input_angle < 0)
    {
        return (input_angle + 360.0f);
    }
    else
    {
        return input_angle;
    }
}

static std::vector<Ranged> extractPointCloud(const fresco_config &config, const pcl::PointCloud<pcl::PointXYZI> &input_cloud)
{
    std::vector<Ranged> cloud;
    cloud.reserve(config.N_SCAN * config.Horizon_SCAN);
    constexpr size_t MAX_NUM_POINTS = 64 * 2250;
    for (auto &&p : input_cloud)
    {
        Ranged point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;

        // point.ring = (k%64) + 1 ;
        cloud.push_back(point);
    }
    // compute azimuthal angle for each point in the cloud
    std::vector<float> azimuth_angle(cloud.size());
    for (int i = 0; i < cloud.size(); i++)
    {
        azimuth_angle[i] = atan2(cloud[i].y, cloud[i].x) / M_PI * 180.0f;
    }

    int32_t ring_idx = -1;
    // drop some points with positive azimuth angle
    // we only start a ring from 0 degree
    if (azimuth_angle[0] > 0)
    {
        ring_idx = 0;
    }
    else
    {
        ring_idx = -1;
    }

    // structed points vector with fixed size of points
    std::vector<Ranged> structured_cloud(config.N_SCAN * config.Horizon_SCAN);
    // seems some compilers have trouble with this type of struct initializer
    // structured_cloud->points.resize(N_SCAN * Horizon_SCAN, pcl::PointXYZIRCT{.intensity = -1});

    // fill row idx and col idx for each point
    float this_azimuth = 0;
    for (int i = 1; i < azimuth_angle.size(); i++)
    {
        // see if new ring arrives
        if (azimuth_angle[i - 1] <= 0 && azimuth_angle[i] > 0)
        {
            ring_idx++;
        }

        // compute column index for this point
        this_azimuth = makeAngleSemiPositive(azimuth_angle[i]);
        int col_idx = static_cast<int>(std::round(this_azimuth / (360.0 / config.Horizon_SCAN)));

        if (ring_idx >= 0 && ring_idx < config.N_SCAN)
        {
            if (col_idx >= config.Horizon_SCAN)
            {
                col_idx = col_idx - config.Horizon_SCAN;
            }
            else if (col_idx < 0)
            {
                col_idx = col_idx + config.Horizon_SCAN;
            }

            cloud[i].row = static_cast<uint16_t>(ring_idx);
            cloud[i].col = static_cast<uint16_t>(col_idx);
            cloud[i].label = static_cast<int16_t>(-2); //-2 means not segmented points
            cloud[i].intensity = -1;

            structured_cloud[ring_idx * config.Horizon_SCAN + col_idx] = cloud[i];
        }
    }

    return structured_cloud;
}

/*
function [img_out] = applyGaussian(img_in)
    sigma = 1;  %设定标准差值，该值越大，滤波效果（模糊）愈明显
    window = double(uint8(3*sigma)*2 + 1);  %设定滤波模板尺寸大小
    %fspecial('gaussian', hsize, sigma)产生滤波掩模
    G = fspecial('gaussian', window, sigma);
    img_out = imfilter(img_in, G, 'conv','replicate','same');
end
*/

static cv::Mat applyGaussian(const cv::Mat &img_in)
{
    cv::Mat img_out;
    cv::GaussianBlur(img_in, img_out, cv::Size(7, 7), 1.0, cv::BORDER_REPLICATE);
    return img_out;
}

static cv::Mat fft2(const cv::Mat &input)
{
    cv::Mat planes[] = {cv::Mat_<float>(input), cv::Mat::zeros(input.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    return complexI;
}

static cv::Mat fftshift(const cv::Mat &input)
{
    cv::Mat output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;
    cv::Mat q0(output, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(output, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(output, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return output;
}

static double interpolate(const cv::Mat &imR, double xR, double yR)
{
    int xf = std::floor(xR);
    int xc = std::ceil(xR);
    int yf = std::floor(yR);
    int yc = std::ceil(yR);
    double v;

    if (xf == xc && yc == yf)
    {
        v = imR.at<double>(xc, yc);
    }
    else if (xf == xc)
    {
        v = imR.at<double>(xf, yf) + (yR - yf) * (imR.at<double>(xf, yc) - imR.at<double>(xf, yf));
    }
    else if (yf == yc)
    {
        v = imR.at<double>(xf, yf) + (xR - xf) * (imR.at<double>(xc, yf) - imR.at<double>(xf, yf));
    }
    else
    {
        cv::Mat A = (cv::Mat_<double>(4, 4) << xf, yf, xf * yf, 1,
                     xf, yc, xf * yc, 1,
                     xc, yf, xc * yf, 1,
                     xc, yc, xc * yc, 1);
        cv::Mat r = (cv::Mat_<double>(4, 1) << imR.at<double>(xf, yf),
                     imR.at<double>(xf, yc),
                     imR.at<double>(xc, yf),
                     imR.at<double>(xc, yc));
        cv::Mat a = A.inv() * r;
        cv::Mat w = (cv::Mat_<double>(1, 4) << xR, yR, xR * yR, 1);
        v = w.dot(a);
    }

    return v;
}
#include <opencv2/opencv.hpp>

float interpolate(const cv::Mat &imR, float xR, float yR)
{
    int xf = std::floor(xR);
    int xc = std::ceil(xR);
    int yf = std::floor(yR);
    int yc = std::ceil(yR);
    float v;

    if (xf == xc && yc == yf)
    {
        v = imR.at<float>(xc, yc);
    }
    else if (xf == xc)
    {
        v = imR.at<float>(xf, yf) + (yR - yf) * (imR.at<float>(xf, yc) - imR.at<float>(xf, yf));
    }
    else if (yf == yc)
    {
        v = imR.at<float>(xf, yf) + (xR - xf) * (imR.at<float>(xc, yf) - imR.at<float>(xf, yf));
    }
    else
    {
        cv::Mat A = (cv::Mat_<float>(4, 4) << xf, yf, xf * yf, 1,
                     xf, yc, xf * yc, 1,
                     xc, yf, xc * yf, 1,
                     xc, yc, xc * yc, 1);
        cv::Mat r = (cv::Mat_<float>(4, 1) << imR.at<float>(xf, yf),
                     imR.at<float>(xf, yc),
                     imR.at<float>(xc, yf),
                     imR.at<float>(xc, yc));
        cv::Mat a = A.inv() * r;
        cv::Mat w = (cv::Mat_<float>(1, 4) << xR, yR, xR * yR, 1);
        v = static_cast<cv::Mat>(w * a).at<float>(0);
    }

    return v;
}

cv::Mat imToPolar(const cv::Mat &imR, float rMin, float rMax, int M, int N)
{
    int Mr = imR.rows;
    int Nr = imR.cols;
    float Om = (Mr + 1) / 2.0f;
    float On = (Nr + 1) / 2.0f;
    float sx = (Mr - 1) / 2.0f;
    float sy = (Nr - 1) / 2.0f;

    cv::Mat imP = cv::Mat::zeros(M, N, CV_32F);

    float delR = (rMax - rMin) / (M - 1);
    float delT = 2 * CV_PI / N;

    for (int ri = 0; ri < M; ri++)
    {
        for (int ti = 0; ti < N; ti++)
        {
            float r = rMin + ri * delR;
            float t = ti * delT;
            float x = r * cos(t);
            float y = r * sin(t);
            float xR = x * sx + Om;
            float yR = y * sy + On;
            imP.at<float>(ri, ti) = interpolate(imR, xR, yR);
        }
    }

    return imP;
}

using vtype = std::array<float, 40>;

vtype ring_key(const cv::Mat &polar_img)
{
    Eigen::MatrixXf ring(1, polar_img.rows);
    Eigen::MatrixXf std(1, polar_img.rows);
    float total = 0;
    for (int i = 0; i < polar_img.rows; i++)
    {
        float sum = 0;
        for (int j = 0; j < polar_img.cols; j++)
        {
            sum += polar_img.at<float>(i, j);
        }
        total += sum;
        ring(0, i) = sum / polar_img.cols;

        sum = 0;
        for (int j = 0; j < polar_img.cols; j++)
        {
            auto val = polar_img.at<float>(i, j) - ring(0, i);
            sum += val * val;
        }
        std(0, i) = std::sqrt(sum / polar_img.rows);
    }

    total /= polar_img.cols * polar_img.rows;
    vtype concat;
    for (int i = 0; i < polar_img.rows; i++)
    {
        concat[i] = ring(0, i) / total;
        concat[i + polar_img.rows] = std(0, i) / total;
    }

    return concat;
}

struct D
{
    cv::Mat fresco;
    vtype key;
};

D gen_descriptor(const fresco_config &config, const std::vector<Ranged> &input_cloud)
{
    std::vector<Ranged> structured_cloud;
    order_cloud(config, input_cloud, structured_cloud);
    mark_ground(config, structured_cloud);

    auto bev = bev_img(structured_cloud);

    cv::Mat resized, merged;
    cv::resize(bev, resized, cv::Size(101, 101));
    cv::Mat input_fft = fft2(resized);
    cv::Mat input_amp = cv::abs(fftshift(input_fft));

    cv::Mat low_freq_part = applyGaussian(input_amp.colRange(31, 71).rowRange(31, 71));
    cv::Mat polar = imToPolar(low_freq_part, 0, 1, 20, 120);
    cv::log(polar, polar);

    auto cell_key = ring_key(polar);

    return {polar, cell_key};
}

/*

%% circle shift to the right
function [img_out] = circleShift(img_in, offset)
    rows = size(img_in, 1);
    cols = size(img_in, 2);

    img_out = zeros(rows, cols);

    for col_idx = 1:cols
        corr_col_idx = col_idx + offset;
        if corr_col_idx > cols
            corr_col_idx = corr_col_idx - cols;
        elseif corr_col_idx <= 0
            corr_col_idx = corr_col_idx + cols;
        end

        img_out(:, col_idx) = img_in(:, corr_col_idx);
    end
end

*/
static cv::Mat circshift(const cv::Mat &_mat, int _num_shift)
{
    // shift columns to right direction
    assert(_num_shift >= 0);

    if (_num_shift == 0)
    {
        cv::Mat shifted_mat = _mat.clone();
        return shifted_mat; // Early return
    }

    cv::Mat shifted_mat = cv::Mat::zeros(_mat.rows, _mat.cols, _mat.type());
    for (int col_idx = 0; col_idx < _mat.cols; col_idx++)
    {
        int new_location = (col_idx + _num_shift) % _mat.cols;
        shifted_mat.col(new_location) = _mat.col(col_idx).clone();
    }

    return shifted_mat;

} // circshift

/*

%% compute L1 dist btw two FreSC
function [dist] = computeL1Dist(img1, img2)
    diff = img1 - img2;
    dist = sum(sum(abs(diff)));
end
*/

float computeL1Dist(const cv::Mat &img1, const cv::Mat &img2)
{
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    return cv::sum(diff)[0];
}

/*
%% compute fresco distance
function [best_offset, fresco_dist] = computeFrescoDist(fresco1, fresco2)
    fresco_dist = 1e10;
    best_offset = 0;
    for angle_offset = 0:59 % since the fft img is symmetrical wrt the center point
        shifted_fresco2 = circleShift(fresco2, angle_offset);
        % log scale seems resulting better angle estimation
        corr = computeL1Dist(log(fresco1), log(shifted_fresco2)); % L1 dist
        if corr < fresco_dist
            fresco_dist = corr;
            best_offset = angle_offset;
        end
    end
end
*/

float fresco_distance(const D &d1, const D &d2, float *yaw)
{
    float fresco_dist = 1.e10;
    int best_offset = 0;
    for (int angle_offset = 0; angle_offset < 60; angle_offset++)
    {
        auto shifted_fresco2 = circshift(d2.fresco, angle_offset);
        float corr = computeL1Dist(d1.fresco, shifted_fresco2);
        if (corr < fresco_dist)
        {
            fresco_dist = corr;
            best_offset = angle_offset;
        }
    }
    *yaw = best_offset;
    return fresco_dist;
}

/*
%% compute cosine distance row-wise
function [dist] = computeCosineDistRowWise(fresco1, fresco2)
    rows = size(fresco1, 1);

    sum = 0;
    for row_idx = 1 : rows
        a = fresco1(row_idx, :);
        b = fresco2(row_idx, :);
        cosine_value = dot(a, b) ./ (norm(a) .* norm(b));
        sum = sum + cosine_value;
    end

    dist = 1.0 - (sum ./ rows);
end
*/

float computeCosineDistRowWise(const cv::Mat &fresco1, const cv::Mat &fresco2)
{
    int rows = fresco1.rows;
    float sum = 0;
    for (int row_idx = 0; row_idx < rows; row_idx++)
    {
        auto a = fresco1.row(row_idx);
        auto b = fresco2.row(row_idx);
        float cosine_value = a.dot(b) / (cv::norm(a) * cv::norm(b));
        sum += cosine_value;
    }
    return 1.0 - (sum / rows);
}

/*

%% compute cosine distance col-wise
function [dist] = computeCosineDistColWise(fresco1, fresco2)
    cols = size(fresco1, 2);

    sum = 0;
    for col_idx = 1 : cols
        a = fresco1(:, col_idx);
        b = fresco2(:, col_idx);
        cosine_value = dot(a, b) ./ (norm(a) .* norm(b));
        sum = sum + cosine_value;
    end

    dist = 1.0 - (sum ./ cols);
end
*/

float computeCosineDistColWise(const cv::Mat &fresco1, const cv::Mat &fresco2)
{
    int cols = fresco1.cols;
    float sum = 0;
    for (int col_idx = 0; col_idx < cols; col_idx++)
    {
        auto a = fresco1.col(col_idx);
        auto b = fresco2.col(col_idx);
        float cosine_value = a.dot(b) / (cv::norm(a) * cv::norm(b));
        sum += cosine_value;
    }
    return 1.0 - (sum / cols);
}

struct fresco_distance_result
{
    size_t index;
    float yaw;
    float fresco_dist;
    float cosine_dist_row_wise;
    float cosine_dist_col_wise;
};

template <typename Point, size_t DIM>
struct KDTree
{
public:
    KDTree()
    {
        // tree = new KDTreeVectorOfVectorsAdaptor<std::vector<Point>, double, DIM, nanoflann::metric_L1>(DIM, values, 10);
    }

    void insert(const Point &point)
    {
        values.push_back(point);
    }

    size_t search_knn(const Point &point, int k, std::vector<size_t> &result_index) const
    {
        result_index.resize(k);
        if (tree == nullptr)
        {
            result_index.clear();
            return 0;
        }
        std::vector<float> out_dists_sqr(k);
        nanoflann::KNNResultSet<float> knnsearch_result(k);         // 预定义
        knnsearch_result.init(&result_index[0], &out_dists_sqr[0]); // 初始化
        return tree->index->knnSearch(point.data(), k, result_index.data(), out_dists_sqr.data(), 10);
    }

    void rebuild_tree()
    {
        if (tree == nullptr)
        {
            tree = new KDTreeVectorOfVectorsAdaptor<std::vector<Point>, float, DIM, nanoflann::metric_L2>(DIM, values, 10);
        }
        else
        {
            tree->index->buildIndex();
        }
        counter = values.size();
    }

    KDTreeVectorOfVectorsAdaptor<std::vector<Point>, float, DIM, nanoflann::metric_L2> *tree = nullptr;
    std::vector<Point> values;
    size_t counter = 0;
};

#include <xsearch.h>

struct fresco_device : search_device
{
    fresco_config config;
    std::vector<D> object_values;
    KDTree<vtype, 40> kdtree;
    std::vector<object_id> joined_id;
};

static constexpr const char *__fresco_module_name = "fresco";

static object_id __fresco_create_object(search_device *object, const pcl::PointCloud<point_type> &cloud)
{
    if (object == nullptr || strcmp(object->module->name, __fresco_module_name) != 0)
    {
        return obj_none;
    }

    auto fresco_object = static_cast<fresco_device *>(object);
    auto r = extractPointCloud(fresco_object->config, cloud);
    auto feature = gen_descriptor(fresco_object->config, r);
    fresco_object->object_values.push_back(std::move(feature));
    return fresco_object->object_values.size() - 1;
}

static size_t __fresco_search(search_device *object, object_id searched_target, search_result *results, size_t max_results)
{
    if (object == nullptr || strcmp(object->module->name, __fresco_module_name) != 0)
    {
        return 0;
    }
    auto fresco_object = static_cast<fresco_device *>(object);

    if (searched_target >= fresco_object->object_values.size())
    {
        return 0;
    }

    if (max_results > fresco_object->kdtree.counter)
    {
        max_results = fresco_object->kdtree.counter;
    }

    if (max_results == 0)
    {
        return 0;
    }

    std::vector<size_t> result_index;
    auto &&point = fresco_object->object_values[searched_target].key;

    size_t candidate_count = max_results < 5 ? 10 : max_results * 2;

    candidate_count = fresco_object->kdtree.search_knn(point, candidate_count, result_index);
    if (candidate_count < max_results)
    {
        max_results = candidate_count;
    }

    std::vector<search_result> result_list(candidate_count);

    for (size_t i = 0; i < candidate_count; i++)
    {
        int id = fresco_object->joined_id[result_index[i]];
        float yaw;
        float distance = fresco_distance(
            fresco_object->object_values[searched_target],
            fresco_object->object_values[id],
            &yaw);

        result_list[i].id = id;
        result_list[i].score = distance;
        result_list[i].yaw = yaw;
    }

    std::sort(result_list.begin(), result_list.end(), [](const search_result &a, const search_result &b)
              { return a.score < b.score; });

    std::copy(result_list.begin(), result_list.begin() + max_results, results);
    return max_results;
}

static bool __fresco_config(search_device *object, const char *key, const char *value)
{
    return false;
}

static void __fresco_join(search_device *object, object_id id)
{
    if (object == nullptr || strcmp(object->module->name, __fresco_module_name) != 0)
    {
        return;
    }

    auto fresco_object = static_cast<fresco_device *>(object);
    fresco_object->joined_id.push_back(id);

    auto v = fresco_object->object_values[id].key;
    fresco_object->kdtree.insert(v);
}

static void __fresco_join_flush(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, __fresco_module_name) != 0)
    {
        return;
    }

    auto fresco_object = static_cast<fresco_device *>(object);
    fresco_object->kdtree.rebuild_tree();
}

static search_device *__fresco_create();
static void __fresco_destroy(search_device *object);

static search_module fresco_module = {
    .name = __fresco_module_name,
    .create_object = &__fresco_create_object,
    .search = &__fresco_search,
    .config = &__fresco_config,
    .join = &__fresco_join,
    .join_flush = &__fresco_join_flush,

    .serialize = nullptr,
    .deserialize = nullptr,
    .save = nullptr,

    .create = &__fresco_create,
    .destroy = &__fresco_destroy,
};

static search_device *__fresco_create()
{
    auto object = new fresco_device();
    object->module = &fresco_module;
    return object;
}

static void __fresco_destroy(search_device *object)
{
    if (object == nullptr || strcmp(object->module->name, __fresco_module_name) != 0)
    {
        return;
    }

    delete static_cast<fresco_device *>(object);
}

register_search_module(fresco_module);
