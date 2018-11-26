#ifndef STREAM_KEYING_HPP
#define STREAM_KEYING_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/svm.h>

#include "keyers/keyer_dnn.hpp"
#include "keyers/keyer_svm.hpp"


struct StreamKeying
{




    const size_t MAX_SAMPLE_UPDATE = 10000; // fg-bg pair
    const int NEAR_REGION_SIZE = 10; // region width of the sure fg/bg in pixel
    const double NEAR_FAR_SAMPLE_RATIO = 0.; // ratio btw near-far regio sample number (0.1 -> 10% far, 90% near)

    StreamKeying()
    {
//        keyer = new KeyerSVM();
        keyer = new KeyerDNN();
    }
    ~StreamKeying() { delete keyer; }

    void update(const cv::Mat_<cv::Vec3b>& origin, const cv::Mat_<uchar>& sureRegions, bool wDebug)
    {
        // ImgSize update/check
        if (imgSize == cv::Size(-1,-1))
            imgSize = origin.size();
        if (imgSize != origin.size())
            throw std::runtime_error("Wrong origin : imgSize!");

        cv::Mat_<uchar> fg, bg;
        cv::threshold(sureRegions, fg, 250, 255, CV_THRESH_BINARY);
        cv::threshold(sureRegions, bg, 10, 255, CV_THRESH_BINARY_INV);

        // Morph?
        // TODO

        // Get interesting area
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*NEAR_REGION_SIZE + 1, 2*NEAR_REGION_SIZE+1 ),
                                                     cv::Point( NEAR_REGION_SIZE, NEAR_REGION_SIZE ) );
        cv::Mat_<uchar> fg_erode, bg_erode;
        cv::erode(fg, fg_erode, element);
        cv::erode(bg, bg_erode, element);
        cv::Mat_<uchar> interestingSureRegion_fg = fg - fg_erode;
        cv::Mat_<uchar> interestingSureRegion_bg = bg - bg_erode;

        // Get pixels
        std::vector<std::pair<cv::Point, cv::Vec3b> > fgColors_near, fgColors_far;
        std::vector<std::pair<cv::Point, cv::Vec3b> > bgColors_near, bgColors_far;
        assert(origin.size() == sureRegions.size());
        for (int r = 0; r < sureRegions.rows; r++)
            for (int c = 0; c < sureRegions.cols; c++)
            {
                const cv::Vec3b& color = origin(r,c);
                if (interestingSureRegion_fg(r,c) == 255)
                    fgColors_near.push_back(std::pair<cv::Point, cv::Vec3b>(cv::Point(c,r), color));
                else if (fg(r,c) == 255)
                    fgColors_far.push_back(std::pair<cv::Point, cv::Vec3b>(cv::Point(c,r), color));
                else if (interestingSureRegion_bg(r,c) == 255)
                    bgColors_near.push_back(std::pair<cv::Point, cv::Vec3b>(cv::Point(c,r), color));
                else if (bg(r,c) == 255)
                    bgColors_far.push_back(std::pair<cv::Point, cv::Vec3b>(cv::Point(c,r), color));
            }
        int fg_near_num = std::min(NEAR_FAR_SAMPLE_RATIO*MAX_SAMPLE_UPDATE, static_cast<double>(fgColors_near.size()));
        int fg_far_num = std::min(MAX_SAMPLE_UPDATE-fg_near_num, fgColors_far.size());
        int bg_near_num = std::min((1.-NEAR_FAR_SAMPLE_RATIO)*MAX_SAMPLE_UPDATE, static_cast<double>(bgColors_near.size()));
        int bg_far_num = std::min(MAX_SAMPLE_UPDATE-bg_near_num, bgColors_far.size());

        // Shuffles
        std::random_shuffle(fgColors_near.begin(), fgColors_near.end());
        std::random_shuffle(fgColors_far.begin(), fgColors_far.end());
        std::random_shuffle(bgColors_near.begin(), bgColors_near.end());
        std::random_shuffle(bgColors_far.begin(), bgColors_far.end());

        // Resize
        fgColors_near.resize(fg_near_num);
        fgColors_far.resize(fg_far_num);
        bgColors_near.resize(bg_near_num);
        bgColors_far.resize(bg_far_num);

        // Copy into one
        std::vector<std::pair<cv::Point, cv::Vec3b> > fgColors = fgColors_near;
        fgColors.insert(fgColors.end(), fgColors_far.begin(), fgColors_far.end());
        std::random_shuffle(fgColors.begin(), fgColors.end());
        std::vector<std::pair<cv::Point, cv::Vec3b> > bgColors = bgColors_near;
        bgColors.insert(bgColors.end(), bgColors_far.begin(), bgColors_far.end());
        std::random_shuffle(bgColors.begin(), bgColors.end());

        KeyerDNN* kDNN = dynamic_cast<KeyerDNN*>(keyer);
        kDNN->updateAll(fgColors, bgColors, origin);

        /*
        // Select points
        cv::Mat_<uchar> selectedFgPts = fg / 2;
        cv::Mat_<uchar> selectedBgPts = bg / 2;
        for (size_t idx = 0; idx < std::min(bgColors.size(), fgColors.size()); idx++)
        {
            const cv::Vec3b& fgColor = fgColors.at(idx).second;
            const cv::Vec3b& bgColor = bgColors.at(idx).second;
            const cv::Point& fgPt = fgColors.at(idx).first;
            const cv::Point& bgPt = bgColors.at(idx).first;

            // Keyer update - BG
            keyer->update(-1., bgColor, bgPt, origin);
            // Keyer update - FG
            keyer->update(+1., fgColor, fgPt, origin);

            // DEBUG
            selectedFgPts(fgPt) = 255;
            selectedBgPts(bgPt) = 255;
        }
        */


        // DEBUG
        if (wDebug)
        {
//            debug_imgs["sureRegions"] = sureRegions;
            debug_imgs["sureFg"] = fg;
            debug_imgs["sureBg"] = bg;
            debug_imgs["interestingSureRegion_fg"] = interestingSureRegion_fg;
            debug_imgs["interestingSureRegion_bg"] = interestingSureRegion_bg;
//            debug_imgs["selectedFgPts"] = selectedFgPts;
//            debug_imgs["selectedBgPts"] = selectedBgPts;
        }
    }

    cv::Mat_<uchar> keying(const cv::Mat_<cv::Vec3b>& origin, const cv::Mat_<uchar>& sureRegions)
    {
//        cv::imshow("asd", origin);
//        cv::waitKey(0);
        cv::Mat_<uchar> key(origin.size());
        for (int r = 0; r < origin.rows; r++)
            for (int c = 0; c < origin.cols; c++)
            {
                const cv::Vec3b& color = origin(r,c);
                cv::Point pos(c,r);
                double label = keyer->decision(color, pos, origin);
//                key(r,c) = (label <= 0.) ? 0 : 255;
                label = std::min(255.,label);
                label = std::max(0.,label);
                key(r,c) = static_cast<uchar>(label);
            }
        return key;
    }

    std::map<std::string, cv::Mat> debug_imgs;

protected:
    Keyer_base* keyer = 0;
    cv::Size imgSize = cv::Size(-1,-1);
};

#endif // STREAM_KEYING_HPP
