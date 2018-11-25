#ifndef FG_BG_SEGMENTATOR_HPP
#define FG_BG_SEGMENTATOR_HPP

#include <set>

#include "sequence_generator.hpp"
#include "utils.hpp"


struct Pixel
{
    const size_t MAX_SAMPLE_SIZE = 64;
    const double UPDATE_PROBABILITY = .5; // bg update probability

    void addBgSample(const cv::Vec3b& color)
    {
        if (bgSamples.size() < MAX_SAMPLE_SIZE)
            bgSamples.push_back(color);
        else
        {
            // Probability check
            const int divider = 10000.;
            double rndNum = rand() % divider;
            if (rndNum / divider > UPDATE_PROBABILITY)
                return;

            // Check exists
            bool exists = false;
            for (size_t i = 0; i < bgSamples.size() && !exists; i++)
                if (bgSamples[i] == color)
                    exists = true;
            if (!exists)
            {
                size_t randomIdx = std::rand() % MAX_SAMPLE_SIZE;
                bgSamples.at(randomIdx) = color;
            }
        }
    }

    bool isBg(const cv::Vec3b& color, double maxDist) const
    {
        bool isBg = false;
        for (size_t i = 0; i < bgSamples.size() && !isBg; i++)
            isBg = calcColorDistance(color, bgSamples[i]) < maxDist;
        return isBg;
    }

    // Neighbours
    Pixel* nl = 0; // left
    Pixel* nr = 0; // right
    Pixel* nt = 0; // top
    Pixel* nb = 0; // bot

    std::vector<cv::Vec3b> bgSamples;
};

struct FgBgSegmentator
{
    void train(const cv::Mat_<cv::Vec3b>& img)
    {
        if (matrix.empty())
        {
            matrixSize = img.size();
            init();
        }

        if (img.size() != matrixSize)
        {
            std::stringstream ss; ss << img.size();
            std::stringstream ss2; ss2 << matrixSize;
            throw std::runtime_error("Different image " + ss.str() + " and matrix " + ss2.str() + " size!");
        }

        for (int r = 0; r < img.rows; r++)
            for (int c = 0; c < img.cols; c++)
            {
                const cv::Vec3b& color = img(r,c);
                matrix[r][c].addBgSample(color);
            }
    }

    cv::Mat_<uchar> segmenting(const cv::Mat_<cv::Vec3b>& img) const
    {
        const double MAX_RGB_DISTANCE_BG = 10;
        const double MAX_RGB_DISTANCE_FG = 50;

        cv::Mat_<uchar> segments = cv::Mat_<uchar>(img.size(), 125);

        for (int r = 0; r < img.rows; r++)
            for (int c = 0; c < img.cols; c++)
            {
                if (matrix[r][c].isBg(img(r,c), MAX_RGB_DISTANCE_BG))   // Sure BG
                    segments(r,c) = 0;
                else if (!matrix[r][c].isBg(img(r,c), MAX_RGB_DISTANCE_FG))   // Sure FG
                    segments(r,c) = 255;
             }
        return segments;
    }

protected:
    void init()
    {
        // Create structure
        for (int r = 0; r < matrixSize.height; r++)
        {
            matrix.push_back(std::vector<Pixel>());
            for (int c = 0; c < matrixSize.width; c++)
                matrix.back().push_back(Pixel());
        }
        // Add neighbours
        for (int r = 0; r < matrixSize.height; r++)
        {
            int tidx = r - 1;
            int bidx = r + 1;
            for (int c = 0; c < matrixSize.width; c++)
            {
                int lidx = c - 1;
                int ridx = c + 1;
                Pixel& pix = matrix[r][c];
                // Set
                if (lidx >= 0) pix.nl = &matrix[r][lidx];
                if (ridx < matrixSize.width) pix.nr = &matrix[r][ridx];
                if (tidx >= 0) pix.nt = &matrix[tidx][c];
                if (bidx < matrixSize.height) pix.nb = &matrix[bidx][c];
            }
        }
        std::cout << "Fg/bg segmentator inited with " << matrixSize << " size\n";
    }

    cv::Size matrixSize;
    std::vector<std::vector<Pixel> > matrix;
};

#endif // FG_BG_SEGMENTATOR_HPP
