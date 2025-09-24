#include "InverseCompositionalKLT.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

int rectHeightDiv2 = 50;
int rectWidthDiv2  = 50;
int fullImgSizeWidth = 512;
int fullImgSizeHeight = 512;
cv::Vec3f speed = {25, 25, 0.15};

cv::Matx23f convertSpeedToAffine() {
    const float cosTheta = cos(speed(2));
    const float sinTheta = sin(speed(2));

    return {
        cosTheta,  sinTheta, speed(0),
        -sinTheta, cosTheta, speed(1)
    };
}

void drawRect(cv::Mat& img, const cv::Point2f& centre) {
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
    cv::Point2f centre = {150, 150};
    drawRect(img, centre);
    cv::Matx23f affine = convertSpeedToAffine();

    InverseCompositionalKLT::InverseCompositionalKLTConfig conf;
    // conf.warpType = InverseCompositionalKLT::WarpType::TRANSLATIONAL;
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