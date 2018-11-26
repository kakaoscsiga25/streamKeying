#ifndef KEYER_SVM_HPP
#define KEYER_SVM_HPP

#include <dlib/svm.h>

#include "keyer_base.hpp"


class KeyerSVM : public Keyer_base
{
    typedef dlib::matrix<double, 5, 1> sample_type;
//    typedef dlib::linear_kernel<sample_type> kernel_type;
//    typedef dlib::polynomial_kernel<sample_type> kernel_type;
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;

public:
    KeyerSVM()
    {
        svm.set_lambda(1e-4);
        svm.set_kernel(kernel_type(50));
        svm.set_max_num_sv(100);
    }

    sample_type dataToFeatVec(const cv::Vec3b& color, const cv::Point& pos, const cv::Size& imgSize)
    {
        assert(imgSize != cv::Size(-1,-1));
        sample_type st;
        st(0) = static_cast<double>(color[0]) / 255.;
        st(1) = static_cast<double>(color[1]) / 255.;
        st(2) = static_cast<double>(color[2]) / 255.;
        st(3) = static_cast<double>(pos.x) / (imgSize.width-1);
        st(4) = static_cast<double>(pos.y) / (imgSize.height-1);
        return st;
    }

    virtual void update(const double& label, const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        sample_type featVec = dataToFeatVec(color, pos, img.size());
        svm.train(featVec, label); // ret with learning rate
    }

    virtual double decision(const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        return svm(dataToFeatVec(color, pos, img.size()));
    }

protected:
    dlib::svm_pegasos<kernel_type> svm;
};

#endif // KEYER_SVM_HPP
