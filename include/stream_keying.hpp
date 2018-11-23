#ifndef STREAM_KEYING_HPP
#define STREAM_KEYING_HPP

#include <opencv2/core.hpp>


struct StreamKeying
{
    void update(const cv::Mat_<cv::Vec3b>& origin, const cv::Mat_<uchar>& sureRegions)
    {
        // TODO
    }

    cv::Mat_<uchar> keying(const cv::Mat_<cv::Vec3b>& origin, const cv::Mat_<uchar>& sureRegions)
    {
        // TODO
        return cv::Mat_<uchar>();
    }

};

#endif // STREAM_KEYING_HPP
