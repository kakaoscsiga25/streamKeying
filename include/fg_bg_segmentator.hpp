#ifndef FG_BG_SEGMENTATOR_HPP
#define FG_BG_SEGMENTATOR_HPP

#include "sequence_generator.hpp"


struct FgBgSegmentator
{
    void train(const cv::Mat_<cv::Vec3b>& img)
    {
        // TODO
    }

    cv::Mat_<uchar> segmenting(const cv::Mat_<cv::Vec3b>& img) const
    {
        // TODO
        return cv::Mat_<uchar>();
    }
};

#endif // FG_BG_SEGMENTATOR_HPP
