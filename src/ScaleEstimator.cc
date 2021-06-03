/**
 *@brief Scale Estimator Class
 *
 */

#include "ScaleEstimator.h"

#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

namespace ORB_SLAM2 {

  /**
 * @brief Construct a new Scale Estimator:: Scale Estimator object
 *
 * @param minDR: minimum distance to start scale estimation
 * @param minKFs:
 * @param scaleDriftThresh: percentage threshold to detect scale drift
 * @param minDist: minimum distance to start scale estimation
 * @param deLimit: absolute threshold for checking scale drtift
 */
  ScaleEstimator::ScaleEstimator(float minDR, int minKFs,
                                 float scaleDriftThresh, float minDist,
                                 float deLimit, int sensor_type)
  {
    minDistDR = minDR;
    mMinKFs = minKFs;
    mScaleDriftThresh = scaleDriftThresh;
    mMinDist = minDist;
    mDELimit = deLimit;
    dr_x_prev = 0;
    dr_y_prev = 0;
    scale = 1.0;
    mSensorType = sensor_type;
    resetParams();
  }

/**
 * @brief reset scale estimator
 *
 */
void ScaleEstimator::resetParams() {
  //cout << "[SE] reset Params" << endl;
  distDR = 0.0;
  distVO = 0.0;
  scale = 1.0;
  count = 0;

  bReady = false;

  mDrHist.clear();
  mVoHist.clear();

  mFirstKFId = 0;
}

/**
 * @brief reset scale estimator
 *
 * @param dr_x: the last x coordinate of DR
 * @param dr_y: the last y coordinate of DR
 * @param t: the last translation of VO
 */
void ScaleEstimator::ScaleEstreset(float dr_x, float dr_y, cv::Mat t) {
  // the last position of DR
  dr_x_prev = dr_x;
  dr_y_prev = dr_y;

  // the last position of VO
  vo_t0 = t.clone();

  resetParams();
}

/**
 * @brief update DR and VO
 *
 * @param dr_x : temporal difference of DR x
 * @param dr_y : temporal difference of DR y
 * @param t : position of VO
 */
void ScaleEstimator::updateDRandVO(float dr_x, float dr_y, cv::Mat t,
                                   float& kfscale) {
  // frame counter
  count++;
  //重新初始化未完成也不用计算scale
  if (count < 3 && mSensorType == 0) {
    dr_x_prev = dr_x;
    dr_y_prev = dr_y;
    //cout << "dr_x_prev, dr_y_prev " << dr_x_prev << ", " << dr_y_prev << endl;
    bReady = false;
    vo_t0 = t.clone();
    return;
  }
  
  //STEREO
  if(count == 1 && mSensorType == 1) 
  { //First Frame do not need to be count
    dr_x_prev = dr_x;
    dr_y_prev = dr_y;
    bReady = false;
    vo_t0 = t.clone();
    return;
  }

  // dist dr
  float DR_del_x = fabs(dr_x - dr_x_prev);
  float DR_del_y = fabs(dr_y - dr_y_prev);
  float dDR = sqrt(DR_del_x * DR_del_x + DR_del_y * DR_del_y);

  // update the last dr
  dr_x_prev = dr_x;
  dr_y_prev = dr_y;

  // dist vo
  double dx = t.at<float>(0) - vo_t0.at<float>(0);
  double dy = t.at<float>(1) - vo_t0.at<float>(1);
  double dz = t.at<float>(2) - vo_t0.at<float>(2);

  // update the last vo
  vo_t0 = t.clone();
  float dVO = sqrt(dx * dx + dy * dy + dz * dz);
  
  // accumulate dr and vo
  distDR += dDR;
  distVO += dVO;
}

/**
 * @brief compute scale factor
 *
 */
void ScaleEstimator::computeScale() { scale = distDR / (distVO + 0.000000001); }

// bool ScaleEstimator::SetLost() {bTrackingLost = true;}

/**
 * @brief get scale factor
 *
 * @return float
 */
float ScaleEstimator::getScale() { return scale; }

/**
 * @brief remember the first KF id of tracking sequence
 *
 * @param id
 */
void ScaleEstimator::setFirstKFId(long unsigned int id) { mFirstKFId = id; }

/**
 * @brief calculate distance between 2 KFs
 *
 * @param pKF1
 * @param pKF2
 * @return float
 */
float distKFs(KeyFrame* pKF1, KeyFrame* pKF2) {
  cv::Mat p1 = pKF1->GetRobotCenter();
  cv::Mat p2 = pKF2->GetRobotCenter();

  float dx = p1.at<float>(0) - p2.at<float>(0);
  float dy = p1.at<float>(1) - p2.at<float>(1);
  float dz = p1.at<float>(2) - p2.at<float>(2);

  return sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief calculate distance between 2 DR positions
 *
 * @param pKF1
 * @param pKF2
 * @return float
 */
float distDRs(const KeyFrame* pKF1, const KeyFrame* pKF2) {
  float dx = (pKF1->mDrX - pKF2->mDrX);
  float dy = (pKF1->mDrY - pKF2->mDrY);

  return sqrt(dx * dx + dy * dy);
}

/**
 * @brief check scale drift
 *
 * @return true
 * @return false
 */
bool ScaleEstimator::checkScaleDrift() {
  // cout << "[SE] ==============================================================="
  //      << endl;
  // cout << "[SE] checkScaleDrift()" << endl;
  // cout << "[SE] mFirstKFId = " << mFirstKFId << endl;

  // need mMinKFs KFs to check scale drift
  if (count < mMinKFs && scale < 7.0) {
    //cout << "[SE] count = " << count << " scale = " << scale << endl;
    return false;
  }

  //cout << "[SE] distDR =  " << distDR << ", distVO = " << distVO << endl;

  // scale factor
  scale = distDR / (distVO + 0.00000001);
  //cout << "[SE] scale factor = " << scale << endl;

  // check scale drift if dr distance > minimum distance
  // relative threshold: scale > mScaleDriftThresh%
  // absolute threshold: |distDR-distVO| > mDELimit (m)
  if (distDR > mMinDist && (scale > (1.0 + mScaleDriftThresh / 100.0) ||
                            scale < (1.0 - mScaleDriftThresh / 100.0) ||
                            fabs(distDR - distVO) > 1.0) ||
      (mCurrentKFID - mLastResizeFrameID > 300 &&  mSensorType == 0) ||
      (scale > 7.0 && distDR > 0.5)) {
    cout << "[SE] Scale drift is found" << endl;
    cout << "[SE] "
            "==============================================================="
         << endl;
    mLastResizeFrameID = mCurrentKFID;
    return true;
  } else {
    cout << "[SE] "
            "==============================================================="
         << endl;
    return false;
  }
}

}  // namespace ORB_SLAM2