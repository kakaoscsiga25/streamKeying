#ifndef KEYER_DNN_HPP
#define KEYER_DNN_HPP

#include "keyer_base.hpp"

struct KeyerDNN : public Keyer_base
{
    virtual void update(const double& label, const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {

    }

    virtual double decision(const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        return 0.;
    }
};

#endif // KEYER_DNN_HPP
