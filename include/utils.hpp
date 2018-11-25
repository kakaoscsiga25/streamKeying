#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>
#include <experimental/filesystem>


void createDirIfNotExist(const std::string& folder)
{
    if (!std::experimental::filesystem::exists(folder))
        std::experimental::filesystem::create_directory(folder);
}

double calcColorDistance(const cv::Vec3b& c1, const cv::Vec3b& c2)
{
    cv::Vec3d diff(c1);
    diff -= c2;
    return std::sqrt(diff.dot(diff));
}

#endif // UTILS_HPP
