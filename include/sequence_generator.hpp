#ifndef SEQUENCE_GENERATOR_HPP
#define SEQUENCE_GENERATOR_HPP

#include <string>
#include <deque>
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


struct SequenceGenerator_base
{
    SequenceGenerator_base(const std::string& PATH) : PATH(PATH) {}
    virtual ~SequenceGenerator_base() {}
    virtual void prepare() = 0;
    virtual cv::Mat_<cv::Vec3b> getNext() { frameID++; return cv::Mat_<cv::Vec3b>(); }
    std::string getFrameIDstring() const { std::stringstream ss; ss << std::setfill('0') << std::setw(5) << frameID; return ss.str(); }

    std::string PATH;
    ulong frameID = 0;
};

struct SequenceGenerator_image : public SequenceGenerator_base
{
    SequenceGenerator_image(const std::string& PATH) : SequenceGenerator_base(PATH) {}
    virtual void prepare()
    {
        // Read images in folder
        for (auto & p : std::experimental::filesystem::directory_iterator(PATH))
        {
            std::stringstream ss;
            ss << p;
            std::string str = ss.str();
            str.erase(str.end()-1); // Remove "
            str.erase(str.begin()); // Remove "
            if (str.size() > 4 && str.substr(str.size()-4, 4) == ".png")
                fileNames.push_back(str);
        }
        // sort by ABC
        std::sort(fileNames.begin(), fileNames.end());

        std::cout << fileNames.size() << " image has found and sorted from " << PATH << "\n";
    }

    virtual cv::Mat_<cv::Vec3b> getNext()
    {
        if (fileNames.empty())
        {
            std::cout << "End of image sequence!\n";
            return cv::Mat_<cv::Vec3b>();
        }

        SequenceGenerator_base::getNext(); // call parent

        // Load image
        cv::Mat img = cv::imread(fileNames.front(), -1);
        fileNames.pop_front();
        cv::Mat_<cv::Vec3b> img_bgr;
        if (img.type() == CV_8UC1)
            cv::cvtColor(img, img_bgr, CV_GRAY2BGR);
        else
            img_bgr = img;
        return  img_bgr;
    }

    std::deque<std::string> fileNames;
};
#include <opencv2/opencv.hpp>
struct SequenceGenerator_video : public SequenceGenerator_base
{
    SequenceGenerator_video(const std::string& PATH) : SequenceGenerator_base(PATH) {}
    ~SequenceGenerator_video() { if (cap.isOpened()) cap.release(); }
    virtual void prepare()
    {
        // File exists?
        // TODO

        // Open the stream
        cap.open(PATH);

        // Check if camera opened successfully
        if(!cap.isOpened())
            throw std::runtime_error("Error opening video stream or file: " + PATH);
    }

    virtual cv::Mat_<cv::Vec3b> getNext()
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "End of video!\n";
            return cv::Mat_<cv::Vec3b>();
        }

        SequenceGenerator_base::getNext(); // call parent
        // TEST
        cv::imshow("asd", frame);
        cv::waitKey(0);
    }
    cv::VideoCapture cap;
};

#endif // SEQUENCE_GENERATOR_HPP
