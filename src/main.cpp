#include <memory>
#include <experimental/filesystem>

#include "sequence_generator.hpp"
#include "fg_bg_segmentator.hpp"
#include "stream_keying.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

int main()
{
    // Read video/image sequence
    const std::string PATH_train = "/home/gergo/data/2018-12-09_keyingVideos/train_usmall";
    const std::string PATH_run = "/home/gergo/data/2018-12-09_keyingVideos/test_small";
//    const std::string PATH_train = "/home/gergo/data/2018-11-24_keyingVideos/train_usmall";
//    const std::string PATH_run = "/home/gergo/data/2018-11-24_keyingVideos/test_small";
//    const std::string PATH_train = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";
//    const std::string PATH_run = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";

    bool wDebug = true;
    std::string debugDir = "./debug";


    // Prepare train
    std::unique_ptr<SequenceGenerator_base> seqGen_train(new SequenceGenerator_image(PATH_train));
//    std::unique_ptr<SequenceGenerator_base> seqGen_train(new SequenceGenerator_video(PATH_train));
    seqGen_train->prepare();
    if (wDebug)
    {
        // Create debug dir if not exist
        if (!std::experimental::filesystem::exists(debugDir))
            std::experimental::filesystem::create_directory(debugDir);
        if (debugDir.back() != '/')
            debugDir += '/';
    }


    /// BG TRAIN
    FgBgSegmentator fgSegm;
    cv::Mat_<cv::Vec3b> img = seqGen_train->getNext();
    std::cout << "Train the fg/bg segmentator" << std::flush;
    while ( !img.empty() )
    {
        std::cout << "." << std::flush;
        fgSegm.train(img);
        img = seqGen_train->getNext();
    }
    std::cout << "DONE\n";


    std::cout << "Prepare for keying...\n";
    /// Run the keying
    // Prepare run
    std::unique_ptr<SequenceGenerator_base> seqGen_run(new SequenceGenerator_image(PATH_run));
    seqGen_run->prepare();

    // Keying
    std::cout << "Keying" << std::flush;
    StreamKeying keyer;
    img = seqGen_run->getNext();
    while ( !img.empty() )
    {
        std::cout << "." << std::flush;

        // Raw fg/bg regmentation
        cv::Mat_<uchar> rawFgBgSegmentedImg = fgSegm.segmenting(img);

        // Create sure Fg/Bg regions
        // TODO
        cv::Mat_<uchar> sureRegions = rawFgBgSegmentedImg;

        cv::Mat_<cv::Vec3b> hackBg = fgSegm.hackBgImg();

        // Update keyer
        keyer.update(img, sureRegions, wDebug, hackBg);

        // Keying
        cv::Mat_<uchar> key = keyer.keying(img, sureRegions);

        if (wDebug)
        {
            std::string saveDir = debugDir + "rawFgBgSegmentedImg/";
            createDirIfNotExist(saveDir);
            cv::imwrite( saveDir + seqGen_run->getFrameIDstring() + ".png", rawFgBgSegmentedImg);
            for (const auto& pair : keyer.debug_imgs)
            {
                saveDir = debugDir + pair.first + "/";
                createDirIfNotExist(saveDir);
                cv::imwrite(saveDir + seqGen_run->getFrameIDstring() + ".png", pair.second);
            }
            saveDir = debugDir + "result/";
            createDirIfNotExist(saveDir);
            cv::imwrite( saveDir + seqGen_run->getFrameIDstring() + ".png", key);
        }

        // Go to next frame
        img = seqGen_run->getNext();
    }
    std::cout << "DONE\n";


    return 0;
}
