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
    std::string debugDir = "./debug";


    // Prepare train
    std::unique_ptr<SequenceGenerator_base> seqGen_train(new SequenceGenerator_image(PATH_train));
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
    std::cout << "Train the fg/bg segmentator...\n";
    while ( !img.empty() )
    {
        std::cerr << "Train " << seqGen_train->getFrameIDstring() << "\n";
        fgSegm.train(img);
        img = seqGen_train->getNext();
    }


    std::cout << "Prepare for keying...\n";
    /// Run the keying
    // Prepare run
    std::unique_ptr<SequenceGenerator_base> seqGen_run(new SequenceGenerator_image(PATH_run));
    seqGen_run->prepare();

    // Keying
    std::cout << "Keying...\n";
    StreamKeying keyer;
    img = seqGen_run->getNext();
    while ( !img.empty() )
    {
        std::cerr << "Run " << seqGen_run->getFrameIDstring() << "\n";

        // Raw fg/bg regmentation
        cv::Mat_<uchar> rawFgBgSegmentedImg = fgSegm.segmenting(img);

        // Create sure Fg/Bg regions
        // TODO
        cv::Mat_<uchar> sureRegions = rawFgBgSegmentedImg;

        // Update keyer
        keyer.update(img, sureRegions, wDebug);

        // Keying
        cv::Mat_<uchar> key = keyer.keying(img, sureRegions);

        if (wDebug)
        {
            cv::imwrite(debugDir + "rawFgBgSegmentedImg_" + seqGen_run->getFrameIDstring() + ".png", rawFgBgSegmentedImg);
            for (const auto& pair : keyer.debug_imgs)
                cv::imwrite(debugDir + pair.first + "_" + seqGen_run->getFrameIDstring() + ".png", pair.second);
        }

        // Go to next frame
        img = seqGen_run->getNext();
    }


    return 0;
}
