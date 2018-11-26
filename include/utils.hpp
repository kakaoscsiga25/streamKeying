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

cv::Mat_<cv::Vec3b> getExtendedROI(const cv::Mat_<cv::Vec3b>& img, const cv::Rect& roi_rect)
{
    cv::Mat_<cv::Vec3b> ROI(roi_rect.height, roi_rect.height);
    for (int r = roi_rect.y; r < roi_rect.y+roi_rect.height; r++)
        for (int c = roi_rect.x; c < roi_rect.x+roi_rect.width; c++)
        {
            int idxr = r;
            if (idxr < 0) idxr = 0;
            if (idxr >= img.rows) idxr = img.rows-1;
            int idxc = c;
            if (idxc < 0) idxc = 0;
            if (idxc >= img.cols) idxc = img.cols-1;
            ROI(r-roi_rect.y, c-roi_rect.x) = img(idxr, idxc);
        }
    return ROI;
}

#endif // UTILS_HPP
