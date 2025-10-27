#include "InverseCompositionalKLT.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

int rectHeightDiv2 = 50;
int rectWidthDiv2  = 50;
int fullImgSizeWidth = 512;
int fullImgSizeHeight = 512;
cv::Vec3f speed = {10, 0, 0.2};
cv::Point2f centre = {150, 150};

cv::Matx23f composeAffineWarp(const cv::Matx23f& affine1, const cv::Matx23f& affine2) {
    const float a1 = affine2(0, 0) * affine1(0, 0) + affine2(1, 0) * affine1(0, 1);
    const float a2 = affine2(0, 1) * affine1(0, 0) + affine2(1, 1) * affine1(0, 1);
    const float a3 = affine2(0, 2) * affine1(0, 0) + affine2(1, 2) * affine1(0, 1) + affine1(0, 2);
    const float a4 = affine2(0, 0) * affine1(1, 0) + affine2(1, 0) * affine1(1, 1);
    const float a5 = affine2(0, 1) * affine1(1, 0) + affine2(1, 1) * affine1(1, 1);
    const float a6 = affine2(0, 2) * affine1(1, 0) + affine2(1, 2) * affine1(1, 1) + affine1(1, 2);
 
    return {
        a1, a2, a3,
        a4, a5, a6
    };
}

cv::Matx23f getNextWarp() {
    cv::Matx23f trans = {
        1, 0, -centre.x,
        0, 1, -centre.y
    };

    const float cosTheta = cos(speed(2));
    const float sinTheta = sin(speed(2));

    cv::Matx23f movement = {
        cosTheta,  sinTheta, speed(0),
        -sinTheta, cosTheta, speed(1)
    };

    cv::Matx23f tmp = composeAffineWarp(movement, trans);
    cv::Matx23f transInv = {
        1, 0, centre.x,
        0, 1, centre.y
    };

    return composeAffineWarp(transInv, tmp);
}

void drawRect(cv::Mat& img) {
    img.create(fullImgSizeHeight, fullImgSizeWidth, CV_8UC1);

    cv::rectangle(
        img,
        {int(centre.x) - rectWidthDiv2, int(centre.y) - rectHeightDiv2},
        {int(centre.x) + rectWidthDiv2, int(centre.y) + rectHeightDiv2},
        cv::Scalar_<unsigned char>(128),
        2
    );

    cv::rectangle(
        img,
        {int(centre.x) + rectWidthDiv2, int(centre.y) - rectHeightDiv2},
        {int(centre.x) + rectWidthDiv2 + 30, int(centre.y) - rectHeightDiv2 - 30},
        cv::Scalar_<unsigned char>(128),
        2
    );
}

void drawPoints(const cv::Mat& img, cv::Mat& imgWCircles, const std::vector<cv::Point2f>& points) {
    img.copyTo(imgWCircles);
    for (const auto& pt: points) {
        cv::circle(
            imgWCircles,
            pt,
            3,
            cv::Scalar_<unsigned char>(255),
            -1
        );
    }
}

int main(int argc, char** argv) {

    cv::Mat img;
    drawRect(img);

    InverseCompositionalKLT::InverseCompositionalKLTConfig conf;
    conf.warpType = InverseCompositionalKLT::WarpType::EUCLIDEAN;
    InverseCompositionalKLT opFlow(conf);

    std::vector<cv::Point2f> pointsToTrackPrevFrame;
    std::vector<cv::Matx23f> warpFnCoeffs;
    std::vector<cv::Point2f> pointsTrackedCurrFrame;
    std::vector<bool> trackedSuccess;
    opFlow.feedFrame(
        img,
        pointsToTrackPrevFrame,
        warpFnCoeffs,
        pointsTrackedCurrFrame,
        trackedSuccess
    );

    pointsToTrackPrevFrame.push_back({centre.x - rectWidthDiv2, centre.y - rectHeightDiv2});
    pointsToTrackPrevFrame.push_back({centre.x - rectWidthDiv2, centre.y + rectHeightDiv2});
    pointsToTrackPrevFrame.push_back({centre.x + rectWidthDiv2, centre.y - rectHeightDiv2});
    pointsToTrackPrevFrame.push_back({centre.x + rectWidthDiv2, centre.y + rectHeightDiv2});

    for (int i = 0; i < 20; i++) {
        cv::Matx23f affine = getNextWarp();
        centre = {
            centre.x * affine(0, 0) + centre.y * affine(0, 1) + affine(0, 2),
            centre.y * affine(1, 0) + centre.y * affine(1, 1) + affine(1, 2)
        };

        cv::Mat viz;
        drawPoints(img, viz, pointsToTrackPrevFrame);
        cv::imshow("viz", viz);
        cv::waitKey();

        cv::warpAffine(
            img,
            img,
            affine,
            img.size()
        );

        opFlow.feedFrame(
            img,
            pointsToTrackPrevFrame,
            warpFnCoeffs,
            pointsTrackedCurrFrame,
            trackedSuccess
        );

        pointsToTrackPrevFrame = pointsTrackedCurrFrame;
        for (const bool& succ: trackedSuccess) {
            spdlog::info("Success: {}", succ);
        }
    }

}