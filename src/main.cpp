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
    const std::string PATH_train = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";
    const std::string PATH_run = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";

    bool wDebug = true;
    const std::string debugDir = "./debug";


    // Prepare train
    std::unique_ptr<SequenceGenerator_base> seqGen(new SequenceGenerator_image(PATH_train));
    seqGen->prepare();
    if (wDebug)
    {
        // Create debug dir if not exist
        if (!std::experimental::filesystem::exists(debugDir))
            std::experimental::filesystem::create_directory(debugDir);
    }


    /// BG TRAIN
    FgBgSegmentator fgSegm;
    cv::Mat_<cv::Vec3b> img = seqGen->getNext();
    std::cout << "Train the fg/bg segmentator...\n";
    while ( !img.empty() )
    {
        fgSegm.train(img);
        img = seqGen->getNext();
    }


    std::cout << "Prepare for keying...\n";
    /// Run the keying
    // Prepare run
    std::unique_ptr<SequenceGenerator_base> seqGen_run(new SequenceGenerator_image(PATH_run));
    seqGen->prepare();

    // Keying
    std::cout << "Keying...\n";
    StreamKeying keyer;
    img = seqGen->getNext();
    while ( !img.empty() )
    {
        // Raw fg/bg regmentation
        cv::Mat_<uchar> rawFgBgSegmentedImg = fgSegm.segmenting(img);

        // Create sure Fg/Bg regions
        // TODO
        cv::Mat_<uchar> sureRegions;

        // Update keyer
        keyer.update(img, sureRegions);

        // Keying
        cv::Mat_<uchar> key = keyer.keying(img, sureRegions);

        if (wDebug)
        {
            cv::imwrite(debugDir + "rawFgBgSegmentedImg_" + seqGen->getFrameIDstring() + ".png", rawFgBgSegmentedImg);
        }

        // Go to next frame
        img = seqGen->getNext();
    }


    return 0;
}
