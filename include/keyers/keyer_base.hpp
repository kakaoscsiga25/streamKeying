#ifndef KEYER_BASE_HPP
#define KEYER_BASE_HPP

#include <opencv2/core.hpp>

struct Keyer_base
{
    virtual void update(const double& label, const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img) = 0;
    virtual double decision(const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img) = 0;
    virtual ~Keyer_base() {}
    virtual std::string info() const { return ""; }
};

#endif // KEYER_BASE_HPP
