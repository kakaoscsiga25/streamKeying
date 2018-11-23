#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>


double calcColorDistance(const cv::Vec3b& c1, const cv::Vec3b& c2)
{
    cv::Vec3d diff(c1);
    diff -= c2;
    return std::sqrt(diff.dot(diff));
}

#endif // UTILS_HPP
