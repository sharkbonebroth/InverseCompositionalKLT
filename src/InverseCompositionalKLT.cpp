#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <Eigen/LU>

#include "InverseCompositionalKLT.hpp"


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
        // Needs to be a clone of the currFrame else the first level will j continue referring to the same block of memory held by currFrame
        cv::buildPyramid(currFrame.clone(), mPrevPyr, mConfig.numPyramidLevels); 
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
        } else if (mConfig.warpType == WarpType::TRANSLATIONAL) {
            runTranslationalKLTSinglePyrLevel(
                mPrevPyr[lv], 
                pyramidPointsPrevFrame, 
                pyrCurr[lv], 
                warpFnCoeffs, 
                trackedSuccess
            );
        } else if (mConfig.warpType == WarpType::AFFINE) {
            runAffineKLTSinglePyrLevel(
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
            pointsTrackedCurrFrame[i] = {
                pointsToTrackPrevFrame[i].x + warpFnCoeffs[i](0, 2),
                pointsToTrackPrevFrame[i].y + warpFnCoeffs[i](1, 2)
            };

            if (
                pointsTrackedCurrFrame[i].x < 0 ||
                pointsTrackedCurrFrame[i].x >= mConfig.inputFrameWidth ||
                pointsTrackedCurrFrame[i].y < 0 ||
                pointsTrackedCurrFrame[i].y >= mConfig.inputFrameHeight
            ) {
                trackedSuccess[i] = false;
                pointsTrackedCurrFrame[i] = {-1, -1};
            }
        } else {
            pointsTrackedCurrFrame[i] = {-1, -1};
        }
    }

    for (int lv = 0; lv < mConfig.numPyramidLevels; lv++) {
        mPrevPyr[lv] = pyrCurr[lv].clone();
    }
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
            
            inverseEuclideanTransform(affineUpdateInv, affineUpdate);
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
    cv::Scharr(img, dxImg, CV_32F, 1, 0);
    cv::Scharr(img, dyImg, CV_32F, 0, 1);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            const int Idx = row * img.cols + col;
            float dx = dxImg.at<float>(row, col);
            float dy = dyImg.at<float>(row, col);

            cv::Matx23f jacobian;
            getEuclideanWarpJacobian(col - mHalfWindowSize, row - mHalfWindowSize, jacobian);

            // Div 32 to normalize the scharr kernel
            steepestDescentImages[Idx] = {
                (jacobian(0, 0) * dx + jacobian(1, 0) * dy) / 32,
                (jacobian(0, 1) * dx + jacobian(1, 1) * dy) / 32,
                (jacobian(0, 2) * dx + jacobian(1, 2) * dy) / 32
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
        cosTheta,  -sinTheta, warpCoeff(0),
        sinTheta, cosTheta, warpCoeff(1)
    };
}

void InverseCompositionalKLT::inverseEuclideanTransform(const cv::Matx23f& euclidian, cv::Matx23f& euclidianInv) const {
    euclidianInv = {
        euclidian(0, 0), euclidian(1, 0), -(euclidian(0, 0) * euclidian(0, 2) + euclidian(1, 0) * euclidian(1, 2)),
        euclidian(0, 1), euclidian(1, 1), -(euclidian(0, 1) * euclidian(0, 2) + euclidian(1, 1) * euclidian(1, 2))
    };
}


void InverseCompositionalKLT::runTranslationalKLTSinglePyrLevel(
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

        std::vector<Eigen::Matrix<float, 2, 1>> steepestDescentImgs;
        getTranslationalSteepestDescentImages(templateROI, steepestDescentImgs);
        Eigen::Matrix<float, 2, 2> hessianInverse;
        // If hessian is degenerate, we do not proceed with tracking
        if (!computeTranslationalHessianInverse(steepestDescentImgs, hessianInverse)) {

            cv::Mat steepestDescentViz0 = cv::Mat(templateROI.size(), CV_32FC1);
            cv::Mat steepestDescentViz1 = cv::Mat(templateROI.size(), CV_32FC1);
            for (int row = 0; row < templateROI.rows; row++) {
                for (int col = 0; col < templateROI.cols; col++) {
                    const int Idx = row * templateROI.cols + col;

                    steepestDescentViz0.at<float>(row, col) = abs(steepestDescentImgs[Idx](0, 0));
                    steepestDescentViz1.at<float>(row, col) = abs(steepestDescentImgs[Idx](1, 0));
                }
            }

            trackedSuccess[i] = false;
            continue;
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

            Eigen::Matrix<float, 2, 1> tmp = Eigen::Matrix<float, 2, 1>::Zero();
            for (int row = 0; row < diff.rows; row++) {
                for (int col = 0; col < diff.cols; col++) {
                    const int idx = diff.cols * row + col;

                    tmp += steepestDescentImgs[idx] * diff.at<float>(row, col);
                }
            }

            Eigen::Matrix<float, 2, 1> deltaCoeff = hessianInverse * tmp;
            if (deltaCoeff.squaredNorm() <= mConfig.epsilon) {
                break;
            }

            warpCoeffs[i](0, 2) -= deltaCoeff(0, 0);
            warpCoeffs[i](1, 2) -= deltaCoeff(1, 0);
        }
    }
}

void InverseCompositionalKLT::getTranslationalWarpJacobian(const float x, const float y, cv::Matx<float, 2, 2>& jacobian) const {
    jacobian = {
        1, 0, 
        0, 1,
    };
}

void  InverseCompositionalKLT::getTranslationalSteepestDescentImages(const cv::Mat& img, std::vector<Eigen::Matrix<float, 2, 1>>& steepestDescentImages) const {
    steepestDescentImages.resize(img.cols * img.rows);

    cv::Mat dxImg, dyImg;
    cv::Scharr(img, dxImg, CV_32F, 1, 0);
    cv::Scharr(img, dyImg, CV_32F, 0, 1);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            const int Idx = row * img.cols + col;
            float dx = dxImg.at<float>(row, col);
            float dy = dyImg.at<float>(row, col);

            cv::Matx22f jacobian;
            getTranslationalWarpJacobian(col, row, jacobian);

            // Div 32 to normalize the scharr kernel
            steepestDescentImages[Idx] = {
                (jacobian(0, 0) * dx + jacobian(1, 0) * dy) / 32,
                (jacobian(0, 1) * dx + jacobian(1, 1) * dy) / 32
            };
        }
    }
}

bool InverseCompositionalKLT::computeTranslationalHessianInverse(
    const std::vector<Eigen::Matrix<float, 2, 1>>& steepestDescentImages, 
    Eigen::Matrix<float, 2, 2>& hessianInverse
) const {
    Eigen::Matrix<float, 2, 2> hessian = Eigen::Matrix<float, 2, 2>::Zero();

    Eigen::Matrix<float, 2, 2> outerProduct;
    for (const auto& sd: steepestDescentImages) {
        outerProduct.noalias() = sd * sd.transpose();
        hessian += outerProduct;
    }

    const float det = hessian.determinant();
    if (det < 0.05) {
        return false;
    }

    hessianInverse = hessian.inverse();

    return true;
}

void InverseCompositionalKLT::runAffineKLTSinglePyrLevel(
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

        std::vector<Eigen::Matrix<float, 6, 1>> steepestDescentImgs;
        getAffineSteepestDescentImages(templateROI, steepestDescentImgs);
        Eigen::Matrix<float, 6, 6> hessianInverse;
        // If hessian is degenerate, we do not proceed with tracking
        if (!computeAffineHessianInverse(steepestDescentImgs, hessianInverse)) {
            trackedSuccess[i] = false;
            continue;
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

            Eigen::Matrix<float, 6, 1> tmp = Eigen::Matrix<float, 6, 1>::Zero();
            for (int row = 0; row < diff.rows; row++) {
                for (int col = 0; col < diff.cols; col++) {
                    const int idx = diff.cols * row + col;
                    tmp += steepestDescentImgs[idx] * diff.at<float>(row, col);
                }
            }

            Eigen::Matrix<float, 6, 1> deltaCoeff = hessianInverse * tmp;
            if (deltaCoeff.squaredNorm() <= mConfig.epsilon) {
                break;
            }

            cv::Matx23f affineUpdateInv = {
                1 + deltaCoeff(0), deltaCoeff(2), deltaCoeff(4),
                deltaCoeff(1), 1 + deltaCoeff(3), deltaCoeff(5)
            };

            cv::Matx23f affineUpdate;
            cv::invertAffineTransform(affineUpdateInv, affineUpdate);

            if (isAffineWarpDegenerate(affineUpdate)) {
                trackedSuccess[i] = false;
                break;
            }

            cv::Matx23f composedAffine;
            composeAffineWarp(warpCoeffs[i], affineUpdate, composedAffine);
            warpCoeffs[i] = composedAffine;
        }
    }

}

void InverseCompositionalKLT::getAffineWarpJacobian(const float x, const float y, cv::Matx<float, 2, 6>& jacobian) const {
    jacobian = {
        x, 0, y, 0,  1, 0,
        0, x, 0, y, 0, 1
    };
}

void InverseCompositionalKLT::getAffineSteepestDescentImages(const cv::Mat& img, std::vector<Eigen::Matrix<float, 6, 1>>& steepestDescentImages) const {
    steepestDescentImages.resize(img.cols * img.rows);

    cv::Mat dxImg, dyImg;
    cv::Scharr(img, dxImg, CV_32F, 1, 0);
    cv::Scharr(img, dyImg, CV_32F, 0, 1);

    cv::Mat tmp0 = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat tmp1 = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat tmp2 = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat tmp3 = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat tmp4 = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat tmp5 = cv::Mat::zeros(img.size(), CV_32FC1);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            const int Idx = row * img.cols + col;
            float dx = dxImg.at<float>(row, col);
            float dy = dyImg.at<float>(row, col);

            cv::Matx<float, 2, 6> jacobian;
            getAffineWarpJacobian(col - mHalfWindowSize, row - mHalfWindowSize, jacobian);

            // Div 32 to normalize the scharr kernel
            steepestDescentImages[Idx] = {
                (jacobian(0, 0) * dx + jacobian(1, 0) * dy) / 32,
                (jacobian(0, 1) * dx + jacobian(1, 1) * dy) / 32,
                (jacobian(0, 2) * dx + jacobian(1, 2) * dy) / 32,
                (jacobian(0, 3) * dx + jacobian(1, 3) * dy) / 32,
                (jacobian(0, 4) * dx + jacobian(1, 4) * dy) / 32,
                (jacobian(0, 5) * dx + jacobian(1, 5) * dy) / 32
            };
            
            tmp0.at<float>(row, col) = (jacobian(0, 0) * dx + jacobian(1, 0) * dy);
            tmp1.at<float>(row, col) = (jacobian(0, 1) * dx + jacobian(1, 1) * dy);
            tmp2.at<float>(row, col) = (jacobian(0, 2) * dx + jacobian(1, 2) * dy);
            tmp3.at<float>(row, col) = (jacobian(0, 3) * dx + jacobian(1, 3) * dy);
            tmp4.at<float>(row, col) = (jacobian(0, 4) * dx + jacobian(1, 4) * dy);
            tmp5.at<float>(row, col) = (jacobian(0, 5) * dx + jacobian(1, 5) * dy);
        }
    }


    // cv::imshow("tmp0", tmp0);
    // cv::imshow("tmp1", tmp1);
    // cv::imshow("tmp2", tmp2);
    // cv::imshow("tmp3", tmp3);
    // cv::imshow("tmp4", tmp4);
    // cv::imshow("tmp5", tmp5);
    // cv::waitKey();
}

bool InverseCompositionalKLT::computeAffineHessianInverse(
    const std::vector<Eigen::Matrix<float, 6, 1>>& steepestDescentImages, 
    Eigen::Matrix<float, 6, 6>& hessianInverse
) const {
    Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();

    Eigen::Matrix<float, 6, 6> outerProduct;
    for (const auto& sd: steepestDescentImages) {
        outerProduct.noalias() = sd * sd.transpose();
        hessian += outerProduct;
    }

    const float det = hessian.determinant();
    if (det < 0.05) {
        spdlog::info("FAIL: {}", det);
        return false;
    }

    hessianInverse = hessian.inverse();

    return true;
}

void InverseCompositionalKLT::sampleWarpedPatch( 
    const cv::Point2f& pt,
    const cv::Mat& img, 
    cv::Matx<float, 2, 3> affine, 
    cv::Mat& patch
) const {
    patch.create(mConfig.windowSize, mConfig.windowSize, CV_8UC1);

    cv::Mat map1 = cv::Mat(mConfig.windowSize, mConfig.windowSize, CV_16SC2);
    cv::Mat map2 = cv::Mat(mConfig.windowSize, mConfig.windowSize, CV_16UC1);

    for (int row = 0; row < mConfig.windowSize; row++) {
        for (int col = 0; col < mConfig.windowSize; col++) {
            float patchX = col - mHalfWindowSize;
            float patchY = row - mHalfWindowSize;

            float preWarpX = patchX + pt.x;
            float preWarpY = patchY + pt.y;

            float deltaX = patchX * affine(0, 0) + patchY * affine(0, 1) + affine(0, 2);
            float deltaY = patchX * affine(1, 0) + patchY * affine(1, 1) + affine(1, 2);

            float destX = pt.x + deltaX;
            float destY = pt.y + deltaY;

            int iDestX = cv::saturate_cast<int>(destX*static_cast<double>(cv::INTER_TAB_SIZE));
            int iDestY = cv::saturate_cast<int>(destY*static_cast<double>(cv::INTER_TAB_SIZE));

            map1.at<cv::Vec2s>(row, col) = {
                static_cast<short>(iDestX >> cv::INTER_BITS),
                static_cast<short>(iDestY >> cv::INTER_BITS)
            };

            map2.at<unsigned short>(row, col) = static_cast<unsigned short>((iDestY & (cv::INTER_TAB_SIZE-1)) *cv::INTER_TAB_SIZE + (iDestX & (cv::INTER_TAB_SIZE-1)));
        }
    }

    cv::remap(
        img,
        patch,
        map1,
        map2,
        cv::INTER_LINEAR
    );

}



void InverseCompositionalKLT::composeAffineWarp(const cv::Matx23f& affine1, const cv::Matx23f& affine2, cv::Matx23f& result) const {
    const float a1 = affine2(0, 0) * affine1(0, 0) + affine2(1, 0) * affine1(0, 1);
    const float a2 = affine2(0, 1) * affine1(0, 0) + affine2(1, 1) * affine1(0, 1);
    const float a3 = affine2(0, 2) * affine1(0, 0) + affine2(1, 2) * affine1(0, 1) + affine1(0, 2);
    const float a4 = affine2(0, 0) * affine1(1, 0) + affine2(1, 0) * affine1(1, 1);
    const float a5 = affine2(0, 1) * affine1(1, 0) + affine2(1, 1) * affine1(1, 1);
    const float a6 = affine2(0, 2) * affine1(1, 0) + affine2(1, 2) * affine1(1, 1) + affine1(1, 2);
 
    result = {
        a1, a2, a3,
        a4, a5, a6
    };
}

bool InverseCompositionalKLT::isAffineWarpDegenerate(const cv::Matx23f& affine) const {
    const float det = (affine(0, 0) * affine(1, 1)) - (affine(0, 1) * affine(1, 0));
    return (det < 0.05);
}