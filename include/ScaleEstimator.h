/**
@brief Scale Estimator Class
*
*/

#ifndef SCALEESTIMATOR_H
#define SCALEESTIMATOR_H

#include <vector>
#include <list>
#include <mutex>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "KeyFrame.h"


using namespace std;


namespace ORB_SLAM2
{


class ScaleEstimator 
{
public:
    ScaleEstimator(float minDR = 1.0, int minKFs = 50, float scaleDriftThresh = 10.0, float minDist = 10.0, float deLimit = 1.0,  int sensor_type = 1);
    ~ScaleEstimator() {}

    // reset all variables including counter and disatances 
    void resetParams();
    // reset all variables with the lates dr and vo results 
    void ScaleEstreset(float dr_x, float dr_y, cv::Mat t);

    // update DR and VO   
    void updateDRandVO(float dr_x, float dr_y, cv::Mat Twc,float &kfscale);


    // calcualte scale factor (dr/vo)
    void computeScale();
    // get scale factor
    float getScale();

    // check scale drift
    bool checkScaleDrift();

    // remeber the first KF id
    void setFirstKFId(long unsigned int id);
   
    int mCurrentKFID;
    int mLastResizeFrameID;
    
    bool mCurrentKFOdomFlag = false;

protected:

private:
    int mSensorType;
    float distDR;   // distance of DR
    float distVO;   // distance of VO
    float scale;    // estimated scale
    int count;      // frame counter
    bool bReady;    //  scale is ready
    float minDistDR; // minimum distance for initialization 
    float dr_x_prev;    // previous dr_x
    float dr_y_prev;    // previous dr_y
    cv::Mat vo_t0;      // previous robot position
    vector<cv::Mat> mDrHist; 
    vector<cv::Mat> mVoHist;
    int mMinKFs;    // minium KFs to check scale drift
    float mScaleDriftThresh; // percentage threshold for scale drift 
    long unsigned int mFirstKFId;   // first KF id after initialization
    float mMinDist;       // minimum distance after reinitialization (m)
    float mDELimit; // absolute distance threshold  (m)
    //bool bTrackingLost; //to set whether tracking is lost

};



} //namespace ORB_SLAM

#endif // SCALEESTIMATOR_H 
