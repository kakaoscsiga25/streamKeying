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

    typedef dlib::decision_function<kernel_type> dec_funct_type;
    typedef dlib::normalized_function<dec_funct_type> funct_type;

public:
    KeyerSVM()
    {
        svm.set_lambda(1e-4);
        svm.set_kernel(kernel_type(5));
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
//        svm.train(featVec, label); // ret with learning rate
        train = true;
        samples.push_back(featVec);
        labels.push_back(label);
    }

    virtual double decision(const cv::Vec3b& color, const cv::Point& pos, const cv::Mat_<cv::Vec3b>& img)
    {
        if (train)
        {
            train = false;

            dlib::svm_c_trainer<kernel_type> trainer;
//            std::cout << "doing cross validation\n";
//            randomize_samples(samples, labels);
//            dlib::vector_normalizer<sample_type> normalizer;
//            normalizer.train(samples);
//            learned_function.normalizer = normalizer;
//            for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
//            {
//                for (double C = 1; C < 100000; C *= 5)
//                {
//                    // tell the trainer the parameters we want to use
//                    trainer.set_kernel(kernel_type(gamma));
//                    trainer.set_c(C);

//                    std::cout << "gamma: " << gamma << "    C: " << C << "   \t";
//                    // Print out the cross validation accuracy for 3-fold cross validation using
//                    // the current gamma and C.  cross_validate_trainer() returns a row vector.
//                    // The first element of the vector is the fraction of +1 training examples
//                    // correctly classified and the second number is the fraction of -1 training
//                    // examples correctly classified.
////                    std::cout << "     cross validation accuracy: "
////                              << cross_validate_trainer(trainer, samples, labels, 5) << "\n";
//                    learned_function.function = trainer.train(samples, labels);
//                    df = trainer.train(samples, labels);
//                    uint ok_pos = 0;
//                    uint ok_neg = 0;
//                    uint sum_pos = 0;
//                    uint sum_neg = 0;
//                    for (uint idx = 0; idx < samples.size(); idx++)
//                    {
//                        double label = learned_function(samples[idx]);
////                        double label = testFunc(samples[idx]);
//                        if (label < 0. && labels[idx] < 0)
//                            ok_neg++;
//                        if (label > 0. && labels[idx] > 0)
//                            ok_pos++;
//                        if (labels[idx] < 0)
//                            sum_neg++;
//                        if (labels[idx] > 0)
//                            sum_pos++;
//                    }
//                    std::cout << double(ok_pos)/sum_pos << " " << double(ok_neg)/sum_neg << "\n" << std::flush;
//                }
//            }
//            throw std::runtime_error("end");

            // Train
            std::cerr << "Train\n";
            trainer.set_kernel(kernel_type(0.78125));
            trainer.set_c(5);

//            dlib::vector_normalizer<sample_type> normalizer;
//            normalizer.train(samples);
//            learned_function.normalizer = normalizer;  // save normalization information
            df = trainer.train(samples, labels);
            samples.clear();
            labels.clear();
        }
        sample_type featVec = dataToFeatVec(color, pos, img.size());
        return df(featVec);
//        return svm(dataToFeatVec(color, pos, img.size()));
    }

    virtual std::string info() const { return "#SV " + std::to_string(svm.get_decision_function().basis_vectors.size()); }

protected:
//    dlib::svm_pegasos<kernel_type> svm;
    dlib::svm_pegasos<kernel_type> svm;
    std::vector<sample_type> samples;
    std::vector<double> labels;
    bool train = true;
    dlib::decision_function<kernel_type> df;

    funct_type learned_function;
};

#endif // KEYER_SVM_HPP
