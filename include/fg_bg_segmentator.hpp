#ifndef FG_BG_SEGMENTATOR_HPP
#define FG_BG_SEGMENTATOR_HPP

#include "sequence_generator.hpp"



struct Pixel
{
    // Neighbours
    Pixel* nl = 0; // left
    Pixel* nr = 0; // right
    Pixel* nt = 0; // top
    Pixel* nb = 0; // bot
};

struct FgBgSegmentator
{
    void train(const cv::Mat_<cv::Vec3b>& img)
    {

    }

    cv::Mat_<uchar> segmenting(const cv::Mat_<cv::Vec3b>& img) const
    {
        // TODO
        return cv::Mat_<uchar>();
    }
};

#endif // FG_BG_SEGMENTATOR_HPP
