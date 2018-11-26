#ifndef KEYER_DNN_HPP
#define KEYER_DNN_HPP

#include "keyer_base.hpp"
#include "MiniDNN.h"
#include "utils.hpp"


class KeyerDNN : public Keyer_base
{
    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::VectorXd Vector;

public:
    KeyerDNN()
    {
        // Create three layers

        // Layer 1 -- convolutional, input size 20x20x1, 3 output channels, filter size 5x5
//        MiniDNN::Layer* layer1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(20, 20, 1, 3, 5, 5);
        // Layer 2 -- max pooling, input size 16x16x3, pooling window size 3x3
//        MiniDNN::Layer* layer2 = new MiniDNN::MaxPooling<MiniDNN::ReLU>(16, 16, 3, 3, 3);
        // Layer 3 -- fully connected, input size 5x5x3, output size 2
//        MiniDNN::Layer* layer3 = new MiniDNN::FullyConnected<MiniDNN::Identity>(5 * 5 * 3, 2);

//        MiniDNN::Layer* layer1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(20, 20, 1, 3, 5, 5);     net.add_layer(layer1);
//        MiniDNN::Layer* layer2 = new MiniDNN::MaxPooling<MiniDNN::ReLU>(16, 16, 3, 3, 3);           net.add_layer(layer2);
        net.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(8, 1, 1, 3, 1, 1));
//        net.add_layer(new MiniDNN::MaxPooling<MiniDNN::ReLU>(8, 1, 3, 1, 1));
//        net.add_layer(new MiniDNN::MaxPooling<MiniDNN::ReLU>(8, 1, 1, 1, 1));
//        net.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(5, 3));
        net.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(8*3, 1));
//        net.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(10, 6));

//        net.set_output(new MiniDNN::BinaryClassEntropy());
//        net.set_output(new MiniDNN::MultiClassEntropy());
        net.set_output(new MiniDNN::RegressionMSE());

        opt.m_lrate = 0.01;


        // DEBUG
        net.set_callback(callback);


        // Initialize parameters with N(0, 0.01^2) using random seed 123
        net.init(0, 0.01, 123);
    }

    Matrix dataToFeatVec(const cv::Vec3b& color, const cv::Point& pos, const cv::Size& imgSize, const cv::Mat_<cv::Vec3b>& img)
    {
        assert(imgSize != cv::Size(-1,-1));
//        cv::Size s(20,20);
//        cv::Rect roiRect(pos.x - s.width/2, pos.y - s.height/2, s.width, s.height);
//        cv::Mat_<cv::Vec3b> ROI = getExtendedROI(img, roiRect);
        Matrix fv = Matrix(8,1);
//        for (int r = 0; r < 400/5; r++)
        {
//            fv(0,0) = static_cast<double>(color[0]);
//            fv(1,0) = static_cast<double>(color[1]);
//            fv(2,0) = static_cast<double>(color[2]);
//            fv(3,0) = static_cast<double>(pos.x);
//            fv(4,0) = static_cast<double>(pos.y);
            fv(0,0) = static_cast<double>(color[0]) / 255.;
            fv(1,0) = static_cast<double>(color[1]) / 255.;
            fv(2,0) = static_cast<double>(color[2]) / 255.;
            fv(3,0) = static_cast<double>(pos.x) / (imgSize.width-1);
            fv(4,0) = static_cast<double>(pos.y) / (imgSize.height-1);
            fv(5,0) = fv(3,0)*fv(3,0);
            fv(7,0) = fv(4,0)*fv(4,0);
            fv(6,0) = fv(3,0)*fv(4,0);
        }
        return fv;
    }

    virtual void update(const double& label, const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        // each column is an observation
        Matrix x = dataToFeatVec(color, pos, img.size(), img);
        Matrix y = Matrix(1, 1);
        y(0,0) = label;

//        std::cerr << x << "\n";
//        std::cerr << y << "\n";

        // Fit the model with a batch size of 100, running 10 epochs with random seed 123
//        net.fit_update(opt, x, y, 10);

        // Iterations on the whole data set
        int epoch = 10;
        static ulong eid = 0;
        for(int k = 0; k < epoch; k++)
        {
//            net.m_callback->m_epoch_id = eid++;

            // Train on each mini-batch
//            for(int i = 0; i < nbatch; i++)
//            int i = 0;
            {
//                net.m_callback->m_batch_id = i;
//                net.m_callback->pre_training_batch(this, x_batches[i], y_batches[i]);

                net.forward(x);
                net.backprop(x, y);
                net.update(opt);

//                net.m_callback->post_training_batch(this, x_batches[i], y_batches[i]);
            }
        }
    }

    void updateAll(std::vector<std::pair<cv::Point, cv::Vec3b> > fgPts, std::vector<std::pair<cv::Point, cv::Vec3b> > bgPts, cv::Mat_<cv::Vec3b> img)
    {
        Matrix x(8, fgPts.size() + bgPts.size());
        Matrix y(1, fgPts.size() + bgPts.size());
        uint idx = 0;
        for (auto pts : fgPts)
        {
            Matrix m = dataToFeatVec(pts.second, pts.first, img.size(), img);
            x.block<8,1>(0,idx) = m;
            y(0,idx) =255;
            idx++;
        }
        for (auto pts : bgPts)
        {
            Matrix m = dataToFeatVec(pts.second, pts.first, img.size(), img);
            x.block<8,1>(0,idx) = m;
            y(0,idx) = 0;
            idx++;
        }
//        std::cerr << y << "\n";
        net.fit(opt, x, y, 100000, 10000, 123);
    }

    virtual double decision(const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        Matrix x = dataToFeatVec(color, pos, img.size(), img);
        Matrix pred = net.predict(x);
//        std::cerr << pred << "\n";
        return  pred(0,0);
    }

    MiniDNN::Network net;
    MiniDNN::RMSProp opt; // optimizer
    MiniDNN::VerboseCallback callback; // DEBUG
};

#endif // KEYER_DNN_HPP
