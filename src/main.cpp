#include <memory>

#include "sequence_generator.hpp"
#include "fg_bg_segmentator.hpp"
#include "stream_keying.hpp"

#include <opencv2/opencv.hpp>

int main()
{
    // Read video/image sequence
    const std::string PATH_train = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";
    const std::string PATH_run = "/home/gergo/data/2018-11-07_chessBoardCalib_noResize_mid";

    bool wDebug = true;


    // Prepare train
    std::unique_ptr<SequenceGenerator_base> seqGen(new SequenceGenerator_image(PATH_train));
    seqGen->prepare();


    /// BG TRAIN
    FgBgSegmentator fgSegm;
    cv::Mat_<cv::Vec3b> img = seqGen->getNext();
    while ( !img.empty() )
    {
        fgSegm.train(img);
        img = seqGen->getNext();
    }


    /// Run the keying
    // Prepare run
    std::unique_ptr<SequenceGenerator_base> seqGen_run(new SequenceGenerator_image(PATH_run));
    seqGen->prepare();

    // Keying
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

        }

        // Go to next frame
        img = seqGen->getNext();
    }


    return 0;
}
