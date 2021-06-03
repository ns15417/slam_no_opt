#ifndef INIT_KEY_FRAME_
#define INIT_KEY_FRAME_
#include "SystemSetting.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include <opencv2/opencv.hpp>
#include "ORBVocabulary.h"


namespace ORB_SLAM2
{

class InitKeyFrame
{

public:

    InitKeyFrame(SystemSetting &SS);
    void UndistortKeyPoints();
    bool PosInGrid(const cv::KeyPoint& kp, int &posX, int &posY);
    void AssignFeaturesToGrid();


    constexpr static int KEYFRAME_GRID_COLS = 64;
    constexpr static int KEYFRAME_GRID_ROWS = 48;
    ORBVocabulary *pVocabulary;
   // KeyFrameDatabase *pKeyFrameDatabase;

    long unsigned int nId;
    long unsigned int mnFrameId;
    double TimeStamp;

    float fGridElementWidthInv;
    float fGridElementHeightInv;
    std::vector<std::size_t> vGrid[KEYFRAME_GRID_COLS][KEYFRAME_GRID_ROWS];

    float fx;
    float fy;
    float cx;
    float cy;
    float invfx;
    float invfy;
    float bf;
    float b;
    float ThDepth;
    int N;

    bool mbDRKF;
    bool mbInitKF;
    float alpha;
    float beta;
    std::vector<cv::KeyPoint> vKps;   //保存当前关键帧锁对应的关键点
    std::vector<cv::KeyPoint> vKpsUn; //equals to vKps for fisheye cuz there is no need to undistort
    std::vector<cv::Point3f> vP3M;

    cv::Mat Descriptors;

    //it's zero for mono
    std::vector<float> vRight;
    std::vector<float> vDepth;

    DBoW2::BowVector BowVec;
    DBoW2::FeatureVector FeatVec;

    int nScaleLevels;
    float fScaleFactor;
    float fLogScaleFactor;
    std::vector<float> vScaleFactors;
    std::vector<float> vLevelSigma2;
    std::vector<float> vInvLevelSigma2;
    std::vector<float> vInvScaleFactors;

    int nMinX;
    int nMinY;
    int nMaxX;
    int nMaxY;
    cv::Mat K;
    cv::Mat DistCoef;

    cv::Mat Tbc;
    //added for DR
    float fDrX;
    float fDrY;
};

} // namespace ORB_SLAM2
#endif