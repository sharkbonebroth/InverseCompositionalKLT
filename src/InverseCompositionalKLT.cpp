#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <Eigen/LU>

#include "InverseCompositionalKLT.hpp"

#include <iostream>

InverseCompositionalKLT::InverseCompositionalKLT(const InverseCompositionalKLTConfig config): 
    mConfig(config),  mHalfWindowSize(config.windowSize / 2) {}

void InverseCompositionalKLT::feedFrame(
    const cv::Mat& currFrame,
    const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
    std::vector<cv::Matx23f>& warpFnCoeffs,
    std::vector<cv::Point2f>& pointsTrackedCurrFrame,
    std::vector<bool>& trackedSuccess
) {
    if (
        currFrame.type() != CV_8UC1
    ) {
        spdlog::error("InverseCompositionalKLT - Incorrect input image type!");
        return;
    }

    if (
        currFrame.cols != mConfig.inputFrameWidth ||
        currFrame.rows != mConfig.inputFrameHeight
    ) {
        spdlog::error(
            "InverseCompositionalKLT - Expected image of size {}, {}. Received: {}, {}",
            mConfig.inputFrameWidth,
            mConfig.inputFrameHeight,
            currFrame.cols,
            currFrame.rows
        );
        return;
    }

    if (mPrevPyr.empty()) {
        cv::buildPyramid(currFrame, mPrevPyr, mConfig.numPyramidLevels);
        return;
    }

    std::vector<cv::Mat> pyrCurr;
    cv::buildPyramid(currFrame, pyrCurr, mConfig.numPyramidLevels);
    
    trackedSuccess.resize(pointsToTrackPrevFrame.size(), true);

    // Initialize warp function coefficients
    const bool priorsProvided = (warpFnCoeffs.size() == pointsToTrackPrevFrame.size());
    if (!priorsProvided) {
        warpFnCoeffs.resize(pointsToTrackPrevFrame.size());

        // Identity affine warp
        for (cv::Matx23f& coeff: warpFnCoeffs) {
            coeff(0, 0) = 1;
            coeff(1, 1) = 1;

            coeff(0, 1) = 0;
            coeff(0, 2) = 0;
            coeff(1, 0) = 0;
            coeff(1, 2) = 0;
        }
    } else {
        // Scale the provided prior so we can start KLT with the coarsest pyramid level
        const float scale = pow(2.0f, -(mConfig.numPyramidLevels - 1));
        for (cv::Matx23f& coeff: warpFnCoeffs) {
            coeff(0, 2) *= scale;
            coeff(1, 2) *= scale;
        }
    }
    
    std::vector<cv::Point2f> pyramidPointsPrevFrame(pointsToTrackPrevFrame.size());
    for (int lv = mConfig.numPyramidLevels-1; lv >= 0; lv--) {
        float scale = pow(2.0f, -lv);
        for (size_t i = 0; i < pointsToTrackPrevFrame.size(); i++) {
            // Rescale point coords in previous frame (previous frame is treated as the template image)
            pyramidPointsPrevFrame[i].x = pointsToTrackPrevFrame[i].x * scale;
            pyramidPointsPrevFrame[i].y = pointsToTrackPrevFrame[i].y * scale;
        }

        if (mConfig.warpType == WarpType::EUCLIDEAN) {
            runEuclideanKLTSinglePyrLevel(
                mPrevPyr[lv], 
                pyramidPointsPrevFrame, 
                pyrCurr[lv], 
                warpFnCoeffs, 
                trackedSuccess
            );
        } else {
            spdlog::error("Unsupported warp type!");
            break;
        }


        if (lv == 0) {
            break;
        }

        // Upscale the warp fn coefficients for the next pyramid level
        for (size_t i = 0; i < pointsToTrackPrevFrame.size(); i++) {
            warpFnCoeffs[i](0, 2) *= 2;
            warpFnCoeffs[i](1, 2) *= 2;
        }

    }

    pointsTrackedCurrFrame.resize(pointsToTrackPrevFrame.size());
    for (size_t i = 0; i < pointsToTrackPrevFrame.size(); i++) {
        if (trackedSuccess[i]) {
            affineWarpPoint(warpFnCoeffs[i], pointsToTrackPrevFrame[i], pointsTrackedCurrFrame[i]);
        } else {
            pointsTrackedCurrFrame[i] = {-1, -1};
        }
    }

    mPrevPyr = std::move(pyrCurr);
}

void InverseCompositionalKLT::sampleWarpedPatch( 
    const cv::Point2f& pt,
    const cv::Mat& img, 
    cv::Matx<float, 2, 3> affine, 
    cv::Mat& patch
) const {
    affine(0, 2) += pt.x - mHalfWindowSize;
    affine(1, 2) += pt.y - mHalfWindowSize;

    cv::warpAffine(
        img, 
        patch, 
        affine, 
        {mConfig.windowSize, mConfig.windowSize}, 
        cv::INTER_LINEAR | cv::BORDER_REPLICATE // should i add WARP_INVERSE_MAP?
    );
}

void InverseCompositionalKLT::runEuclideanKLTSinglePyrLevel(
    const cv::Mat& prevFrame,
    const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
    const cv::Mat& currFrame,
    std::vector<cv::Matx23f>& warpCoeffs,
    std::vector<bool>& trackedSuccess
) const {
    for (size_t i = 0; i < pointsToTrackPrevFrame.size(); i++) {
        if (!trackedSuccess[i]) {
            continue;
        }

        cv::Mat templateROI;
        cv::getRectSubPix(
            prevFrame, 
            {mConfig.windowSize, mConfig.windowSize}, 
            pointsToTrackPrevFrame[i], 
            templateROI,
            CV_32F
        );

        std::vector<Eigen::Matrix<float, 3, 1>> steepestDescentImgs;
        getEuclideanSteepestDescentImages(templateROI, steepestDescentImgs);
        Eigen::Matrix<float, 3, 3> hessianInverse;
        // If hessian is degenerate, we do not proceed with tracking
        if (!computeEuclideanHessianInverse(steepestDescentImgs, hessianInverse)) {
            trackedSuccess[i] = false;
            continue;
        } else {
            spdlog::info("FFF");
        }

        for (int iterNum = 0; iterNum < mConfig.maxIterations; iterNum++) {
            cv::Mat currROI;
            sampleWarpedPatch(
                pointsToTrackPrevFrame[i], 
                currFrame, 
                warpCoeffs[i], 
                currROI
            );

            cv::Mat diff;
            cv::subtract(currROI, templateROI, diff, cv::noArray(), CV_32FC1);

            Eigen::Matrix<float, 3, 1> tmp = Eigen::Matrix<float, 3, 1>::Zero();
            for (int row = 0; row < diff.rows; row++) {
                for (int col = 0; col < diff.cols; col++) {
                    const int idx = diff.cols * row + col;

                    tmp += steepestDescentImgs[idx] * diff.at<float>(row, col);
                }
            }

            Eigen::Matrix<float, 3, 1> deltaCoeff = hessianInverse * tmp;
            if (deltaCoeff.squaredNorm() <= mConfig.epsilon) {
                break;
            }

            cv::Matx23f affineUpdateInv, affineUpdate, composedAffine;
            convertEuclideanWarpCoeffToAffine(deltaCoeff, affineUpdateInv);
            if (isAffineWarpDegenerate(affineUpdateInv)) {
                trackedSuccess[i] = false;
                break;
            }
            cv::invertAffineTransform(affineUpdateInv, affineUpdate);
            composeAffineWarp(warpCoeffs[i], affineUpdate, composedAffine);
            warpCoeffs[i] = composedAffine;
        }
    }
}

void InverseCompositionalKLT::getEuclideanWarpJacobian(const float x, const float y, cv::Matx<float, 2, 3>& jacobian) const {
    jacobian = {
        1, 0, -y,
        0, 1, x
    };
}

void  InverseCompositionalKLT::getEuclideanSteepestDescentImages(const cv::Mat& img, std::vector<Eigen::Matrix<float, 3, 1>>& steepestDescentImages) const {
    steepestDescentImages.resize(img.cols * img.rows);

    cv::Mat dxImg, dyImg;
    cv::Scharr(img, dxImg, CV_16U, 1, 0);
    cv::Scharr(img, dyImg, CV_16U, 0, 1);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            const int Idx = row * img.cols + col;
            float dx = static_cast<float>(dxImg.at<short>(row, col));
            float dy = static_cast<float>(dyImg.at<short>(row, col));

            cv::Matx23f jacobian;
            getEuclideanWarpJacobian(col, row, jacobian);
            steepestDescentImages[Idx] = {
                jacobian(0, 0) * dx + jacobian(0, 1) * dy,
                jacobian(1, 0) * dx + jacobian(1, 1) * dy,
                jacobian(2, 0) * dx + jacobian(2, 1) * dy
            };
        }
    }

    cv::Mat steepestDescentViz = cv::Mat(img.size(), CV_8UC3);
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            const int Idx = row * img.cols + col;

            steepestDescentViz.at<cv::Vec3b>(row, col) = {
                static_cast<unsigned char>(steepestDescentImages[Idx](0, 0)),
                static_cast<unsigned char>(steepestDescentImages[Idx](1, 0)),
                static_cast<unsigned char>(steepestDescentImages[Idx](2, 0))
            };
        }
    }

}

bool InverseCompositionalKLT::computeEuclideanHessianInverse(
    const std::vector<Eigen::Matrix<float, 3, 1>>& steepestDescentImages,
    Eigen::Matrix<float, 3, 3>& hessianInverse
) const {
    Eigen::Matrix<float, 3, 3> hessian = Eigen::Matrix<float, 3, 3>::Zero();

    Eigen::Matrix<float, 3, 3> outerProduct;
    for (const auto& sd: steepestDescentImages) {
        outerProduct.noalias() = sd * sd.transpose();
        hessian += outerProduct;
    }

    std::cout << hessian << std::endl;
    const float det = hessian.determinant();
    if (det < 0.05) {
        return false;
    }

    hessianInverse = hessian.inverse();

    return true;
}

void InverseCompositionalKLT::convertEuclideanWarpCoeffToAffine(Eigen::Matrix<float, 3, 1> warpCoeff, cv::Matx23f& affine) const {
    const float cosTheta = cos(warpCoeff(2));
    const float sinTheta = sin(warpCoeff(2));

    affine = {
        cosTheta,  sinTheta, warpCoeff(0),
        -sinTheta, cosTheta, warpCoeff(1)
    };
}

void InverseCompositionalKLT::getTranslationalWarpJacobian(const float x, const float y, cv::Matx<float, 2, 2>& jacobian) const {
    jacobian = {
        1, 0, 
        0, 1,
    };
}

void InverseCompositionalKLT::getAffineWarpJacobian(const float x, const float y, cv::Matx<float, 2, 6>& jacobian) const {
    jacobian = {
        x, 0, y, 0,  1, 0,
        0, x, 0, y, 0, 1
    };
}


void InverseCompositionalKLT::composeAffineWarp(const cv::Matx23f& affine1, const cv::Matx23f& affine2, cv::Matx23f& result) const {
    // Eigen::Matrix<float, 3, 3> affine1Eig, affine2Eig, composedEig;

    // affine1Eig  <<  affine1(0, 0), affine1(0, 1), affine1(0, 2),
    //                 affine1(1, 0), affine1(1, 1), affine1(1, 2),
    //                 0, 0, 1;

    // affine2Eig  <<  affine2(0, 0), affine2(0, 1), affine2(0, 2),
    //                 affine2(1, 0), affine2(1, 1), affine2(1, 2),
    //                 0, 0, 1;

    // composedEig.noalias() = affine1Eig * affine2Eig;
    
    // result = {
    //     composedEig(0, 0), composedEig(0, 1), composedEig(0, 2),
    //     composedEig(1, 0), composedEig(1, 1), composedEig(1, 2)
    // };

    const float a1 = affine2(0, 0) * affine1(0, 0) + affine2(1, 0) * affine1(0, 1) + affine2(2, 0) * affine1(0, 2);
    const float a2 = affine2(0, 1) * affine1(0, 0) + affine2(1, 1) * affine1(0, 1) + affine2(2, 1) * affine1(0, 2);
    const float a3 = affine2(0, 2) * affine1(0, 0) + affine2(1, 2) * affine1(0, 1) + affine2(2, 2) * affine1(0, 2);
    const float a4 = affine2(0, 0) * affine1(1, 0) + affine2(1, 0) * affine1(1, 1) + affine2(2, 0) * affine1(1, 2);
    const float a5 = affine2(0, 1) * affine1(1, 0) + affine2(1, 1) * affine1(1, 1) + affine2(2, 1) * affine1(1, 2);
    const float a6 = affine2(0, 2) * affine1(1, 0) + affine2(1, 2) * affine1(1, 1) + affine2(2, 2) * affine1(1, 2);

    result = {
        a1, a2, a3,
        a4, a5, a6
    };
}

bool InverseCompositionalKLT::isAffineWarpDegenerate(const cv::Matx23f& affine) const {
    const float det = (affine(0, 0) * affine(1, 1)) - (affine(0, 1) * affine(1, 0));
    return (det < 0.05);
}

void InverseCompositionalKLT::affineWarpPoint(const cv::Matx23f& affine, const cv::Point2f& point, cv::Point2f& warpedPoint) const {
    warpedPoint = {
        point.x * affine(0, 0) + point.y * affine(0, 1) + affine(0, 2),
        point.x * affine(1, 0) + point.y * affine(1, 1) + affine(1, 2)
    };
}
