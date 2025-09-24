#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <Eigen/Core>


class InverseCompositionalKLT {

public:
    enum class WarpType {
        EUCLIDEAN,
        TRANSLATIONAL,
        AFFINE
    };

    struct InverseCompositionalKLTConfig {
        int windowSize = 21;
        int maxIterations = 30; 
        int numPyramidLevels = 3;
        float epsilon = 0.01; // Min delta for early optimizer termination
        float minEigThreshold = 0.001; 
        int inputFrameWidth = 512;
        int inputFrameHeight = 512;
        WarpType warpType = WarpType::EUCLIDEAN;
    };

public:
    InverseCompositionalKLT(const InverseCompositionalKLTConfig config);

    /**
    * @param currFrame                  Image in which to search for points in the cached previous frame (if a previous frame has been fed)
    * @param pointsToTrackPrevFrame     Vector of points in the previous frame to track
    * @param warpFnCoeffs               Warp function priors provided as 2x3 affine matrices
    *                                   If populated, these will be used as the initialization point for optimization. 
    *                                   The final warp function coefficients will be populated here after the algorithm runs.
    * @param pointsTrackedCurrFrame     Tracked points in currFrame
    * @param trackedSuccess             Vector of bools indicating tracking success for each input point in pointsToTrack
    */
    void feedFrame(
        const cv::Mat& currFrame,
        const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
        std::vector<cv::Matx23f>& warpFnCoeffs,
        std::vector<cv::Point2f>& pointsTrackedCurrFrame,
        std::vector<bool>& trackedSuccess
    );

protected:
    void sampleWarpedPatch( // Given a point for a patch in the template image, samples the corresponding patch in the warped new image, given the estimated affine warp
        const cv::Point2f& pt,
        const cv::Mat& img, 
        cv::Matx<float, 2, 3> affine, 
        cv::Mat& patch
    ) const;

    void runEuclideanKLTSinglePyrLevel(
        const cv::Mat& prevFrame,
        const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
        const cv::Mat& currFrame,
        std::vector<cv::Matx23f>& warpCoeffs,
        std::vector<bool>& trackedSuccess
    ) const;
    void getEuclideanWarpJacobian(const float x, const float y, cv::Matx<float, 2, 3>& jacobian) const; // Assumes that theta is counterclockwise positive
    void getEuclideanSteepestDescentImages(const cv::Mat& img, std::vector<Eigen::Matrix<float, 3, 1>>& steepestDescentImages) const;
    bool computeEuclideanHessianInverse(const std::vector<Eigen::Matrix<float, 3, 1>>& steepestDescentImages, Eigen::Matrix<float, 3, 3>& hessianInverse) const;
    void convertEuclideanWarpCoeffToAffine(Eigen::Matrix<float, 3, 1> warpCoeff, cv::Matx23f& affine) const;
    void inverseEuclideanTransform(const cv::Matx23f& euclidian, cv::Matx23f& euclidianInv) const;

    void runTranslationalKLTSinglePyrLevel(
        const cv::Mat& prevFrame,
        const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
        const cv::Mat& currFrame,
        std::vector<cv::Matx23f>& warpCoeffs,
        std::vector<bool>& trackedSuccess
    ) const;
    void getTranslationalWarpJacobian(const float x, const float y, cv::Matx<float, 2, 2>& jacobian) const;
    void getTranslationalSteepestDescentImages(const cv::Mat& img, std::vector<Eigen::Matrix<float, 2, 1>>& steepestDescentImages) const;
    bool computeTranslationalHessianInverse(const std::vector<Eigen::Matrix<float, 2, 1>>& steepestDescentImages, Eigen::Matrix<float, 2, 2>& hessianInverse) const;

    void runAffineKLTSinglePyrLevel(
        const cv::Mat& prevFrame,
        const std::vector<cv::Point2f>& pointsToTrackPrevFrame,
        const cv::Mat& currFrame,
        std::vector<cv::Matx23f>& warpCoeffs,
        std::vector<bool>& trackedSuccess
    ) const;
    void getAffineWarpJacobian(const float x, const float y, cv::Matx<float, 2, 6>& jacobian) const;

    void composeAffineWarp(const cv::Matx23f& affine1, const cv::Matx23f& affine2, cv::Matx23f& result) const;
    bool isAffineWarpDegenerate(const cv::Matx23f& affine) const;

protected:
    InverseCompositionalKLTConfig mConfig;
    const int mHalfWindowSize;

    std::vector<cv::Mat> mPrevPyr;
};