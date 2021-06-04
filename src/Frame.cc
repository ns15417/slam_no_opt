/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Frame.h"

#include <cmath>
#include <thread>
#include "CameraModels/GeometricCamera.h"
#include "Converter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame() {}

// Copy Constructor
Frame::Frame(const Frame &frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpORBextractorLeft(frame.mpORBextractorLeft),
      mpORBextractorRight(frame.mpORBextractorRight),
      mTimeStamp(frame.mTimeStamp),
      mK(frame.mK.clone()),
      mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf),
      mb(frame.mb),
      mThDepth(frame.mThDepth),
      N(frame.N),
      mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight),
      mvKeysUn(frame.mvKeysUn),
      mvuRight(frame.mvuRight),
      mvDepth(frame.mvDepth),
      mvP3M(frame.mvP3M),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()),
      mDescriptorsRight(frame.mDescriptorsRight.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2),
      mDrX(frame.mDrX),
      mDrY(frame.mDrY),
      mOdomFlag(frame.mOdomFlag),
      mSensor(frame.mSensor),
      mFrameTbc(frame.mFrameTbc),
      mpCamera(frame.mpCamera),
      mpCamera2(frame.mpCamera2) {
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++) mGrid[i][j] = frame.mGrid[i][j];

  if (!frame.mTcw.empty()) SetPose(frame.mTcw);
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
             const double &timeStamp, ORBextractor *extractorLeft,
             ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, cv::Mat &Rrl,
             cv::Mat &trinl, const float &bf, const float &thDepth,
             GeometricCamera *pCamera, GeometricCamera *pCamera2,
             float dr_x, float dr_y, bool odom_flag,
             int sensor_type, cv::Mat Tbc)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractorLeft),
      mpORBextractorRight(extractorRight),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mRrl(Rrl.clone()),
      mtlinr(trinl.clone()),
      mbf(bf),
      mThDepth(thDepth),
      mpReferenceKF(static_cast<KeyFrame *>(NULL)),
      mSensor(sensor_type),
      mFrameTbc(Tbc),
      mpCamera(pCamera),
      mpCamera2(pCamera2)
{
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
  thread threadRight(&Frame::ExtractORB, this, 1, imRight);
  threadLeft.join();
  threadRight.join();

  N = mvKeys.size();
  NRight = mvKeysRight.size();
  std::cout<< "Keypoint in left camera: " << N << " Keypoints in right camera " << NRight << std::endl;

  mDrX = dr_x;
  mDrY = dr_y;
  mOdomFlag = odom_flag;

  if (mvKeys.empty()) return;

  image_width = imLeft.cols;
  image_height= imLeft.rows;

  UndistortKeyPoints();
  UndistortKeyPointsRight();

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imLeft);

    mfGridElementWidthInv =
        static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
    mfGridElementHeightInv =
        static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
  AssignFeaturesToGridRight();
  ComputeFisheyeStereoMatches();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth,
             const double &timeStamp, ORBextractor *extractor,
             ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
             const float &thDepth, int sensor_type,
             cv::Mat Tbc)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth),
      mSensor(sensor_type),
      mFrameTbc(Tbc)
{
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) return;

  UndistortKeyPoints();

  ComputeStereoFromRGBD(imDepth);

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const double &timeStamp,
             ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, const float &bf, const float &thDepth,
             GeometricCamera *pCamera, float dr_x, float dr_y, bool odom_flag, int sensor_type,
             cv::Mat Tbc)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth),
      mSensor(sensor_type),
      mFrameTbc(Tbc),
      mpCamera(pCamera) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  mDrX = dr_x;
  mDrY = dr_y;
  mOdomFlag = odom_flag;

  // ORB extraction
  ExtractORB(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) return;

  UndistortKeyPoints();

  // Set no stereo information
  mvuRight = vector<float>(N, -1);
  mvDepth = vector<float>(N, -1);

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
}

Frame::Frame(long unsigned int i) {
  mnId = i;
  nNextId = i + 1;
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

void Frame::AssignFeaturesToGridRight() {
  int NR = mvKeysRight.size();
  int nReserve = 0.5f * NR / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGridRight[i][j].reserve(nReserve);
  }

  for (int i = 0; i < NR; i++) {
    const cv::KeyPoint &kpright = mvKeysRight[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kpright, nGridPosX, nGridPosY))
      mGridRight[nGridPosX][nGridPosY].push_back(i);
  }
}

void Frame::ExtractORB(int flag, const cv::Mat &im) {
  vector<int> vLapping = {100, 600};
  if (flag == 0)
    monoLeft =
        (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
  else
    monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight,
                                       mDescriptorsRight, vLapping);
}

void Frame::SetPose(cv::Mat Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() {
  mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0, 3).col(3);
  mOw = -mRcw.t() * mtcw;

  cv::Mat tcw = mtcw.clone();
  cv::Mat baselink_in_c(3, 1, CV_32F);
  cv::Mat Rbc(3, 3, CV_32F);
  cv::Mat tbc(3, 1, CV_32F);
  mFrameTbc.colRange(0, 3).rowRange(0, 3).copyTo(Rbc);
  mFrameTbc.col(3).rowRange(0, 3).copyTo(tbc);

  baselink_in_c = -Rbc.t() * tbc;

  cv::Mat baselink_in_world = mRwc * baselink_in_c + mOw;
  cv::Mat baselink_in_map = Rbc * baselink_in_world + tbc;

  mRobotCw = baselink_in_map.clone();
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f) return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX) return false;
  if (v < mnMinY || v > mnMaxY) return false;

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);

  if (dist < minDistance || dist > maxDistance) return false;

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit) return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjXR = u - mbf * invz;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

bool Frame::isInFrustumFisheye(MapPoint *pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Project in image and check it is not outside
  const float alpha = mDistCoef.at<float>(0);
  const float beta = mDistCoef.at<float>(1);
  const float imd = sqrt(beta * (PcX * PcX + PcY * PcY) + PcZ * PcZ);

  const float mx = PcX / (alpha * imd + (1 - alpha) * PcZ);
  const float my = PcY / (alpha * imd + (1 - alpha) * PcZ);

  const float mR2 = mx * mx + my * my;
  const float mR2range = 1.0f / (beta * (2 * alpha - 1));

  if (mR2 > mR2range) return false;

  const float u = fx * mx + cx;
  const float v = fy * my + cy;

  if (u < mnMinX || u > mnMaxX) return false;
  if (v < mnMinY || v > mnMaxY) return false;

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);

  if (dist < minDistance || dist > maxDistance) return false;

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  // if(viewCos<viewingCosLimit)
  //    return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  // pMP->mTrackProjXR = u - mbf*invz;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y,
                                        const float &r, const int minLevel,
                                        const int maxLevel) const {
  vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX =
      max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS) return vIndices;

  const int nMaxCellX =
      min((int)FRAME_GRID_COLS - 1,
          (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) return vIndices;

  const int nMinCellY =
      max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS) return vIndices;

  const int nMaxCellY =
      min((int)FRAME_GRID_ROWS - 1,
          (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty()) continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
        if (bCheckLevels) {
          if (kpUn.octave < minLevel) continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel) continue;
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  // Keypoint's coordinates are undistorted, which could cause to go out of the
  // image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
      posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

vector<size_t> Frame::GetFeaturesInAreaRight(const float &x, const float &y,
                                             const float &r, const int minLevel,
                                             const int maxLevel) const {
  vector<size_t> vIndices;
  vIndices.reserve(N);
  // 已经将特征点分配到(40/3, 10)的各自里， 总共为48*48个，
  // 所以这里主要是计算起始和结束的格子数；
  const int nMinCellX =
      max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS) return vIndices;

  const int nMaxCellX =
      min((int)FRAME_GRID_COLS - 1,
          (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) return vIndices;

  const int nMinCellY =
      max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS) return vIndices;

  const int nMaxCellY =
      min((int)FRAME_GRID_ROWS - 1,
          (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = mGridRight[ix][iy];
      if (vCell.empty())  // 当前cell中是都有特征点
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpRight = mvKeysRight[vCell[j]];
        if (bCheckLevels) {
          if (kpRight.octave < minLevel) continue;
          if (maxLevel >= 0)
            if (kpRight.octave > maxLevel) continue;
        }

        const float distx = kpRight.pt.x - x;
        const float disty = kpRight.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

void Frame::ComputeBoW() {
  if (mBowVec.empty()) {
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::UndistortKeyPoints() {
    if (mDistCoef.at<float>(0) == 0.0) {
    mvKeysUn = mvKeys;
    return;
  }
  // Fill M, EUCM p=KM
  mvKeysUn.resize(N);
  mvP3M.resize(N);
  for (int i = 0; i < N; i++) {
    cv::Point3f P3M = mpCamera->Img2Camera(mvKeys[i].pt);
    mvP3M[i].x = P3M.x;
    mvP3M[i].y = P3M.y;
    mvP3M[i].z = P3M.z;

    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mvKeys[i].pt.x;
    kp.pt.y = mvKeys[i].pt.y;
    mvKeysUn[i] = kp;
  }
}

void Frame::UndistortKeyPointsRight() {
  if (mpCamera2->mvParameters[4] == 0.0) {
    mvKeysRightUn = mvKeysRight;
    return;
  }
  std::cout << "mvKeysRight size is " << mvKeysRight.size() << endl;
  mvKeysRightUn.resize(NRight);
  mvP3MRight.resize(NRight);
  for (int i = 0; i < NRight; i++) {
    cv::Point3f P3M = mpCamera2->Img2Camera(mvKeysRight[i].pt);
    mvP3MRight[i].x = P3M.x;
    mvP3MRight[i].y = P3M.y;
    mvP3MRight[i].z = P3M.z;

    cv::KeyPoint kp = mvKeysRight[i];
    kp.pt.x = mvKeysRight[i].pt.x;
    kp.pt.y = mvKeysRight[i].pt.y;
    mvKeysRightUn[i] = kp;
  }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
  /*if(mDistCoef.at<float>(0)!=0.0)
  {
      cv::Mat mat(4,2,CV_32F);
      mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
      mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
      mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
      mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

      // Undistort corners
      mat=mat.reshape(2);
      cv::undistortPoints(mat,mat,mK,cv::Mat(),cv::Mat(),mK);
      mat=mat.reshape(1);

      mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
      mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
      mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
      mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

  }
  else*/
  {
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

void Frame::ComputeStereoMatches() {
  mvuRight = vector<float>(N, -1.0f);
  mvDepth = vector<float>(N, -1.0f);

  const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

  const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

  // Assign keypoints to row table
  vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

  for (int i = 0; i < nRows; i++) vRowIndices[i].reserve(200);

  const int Nr = mvKeysRight.size();

  for (int iR = 0; iR < Nr; iR++) {
    const cv::KeyPoint &kp = mvKeysRight[iR];
    const float &kpY = kp.pt.y;
    const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
    const int maxr = ceil(kpY + r);
    const int minr = floor(kpY - r);

    for (int yi = minr; yi <= maxr; yi++) vRowIndices[yi].push_back(iR);
  }

  // Set limits for search
  const float minZ = mb;
  const float minD = 0;
  const float maxD = mbf / minZ;

  // For each left keypoint search a match in the right image
  vector<pair<int, int> > vDistIdx;
  vDistIdx.reserve(N);

  for (int iL = 0; iL < N; iL++) {
    const cv::KeyPoint &kpL = mvKeys[iL];
    const int &levelL = kpL.octave;
    const float &vL = kpL.pt.y;
    const float &uL = kpL.pt.x;

    const vector<size_t> &vCandidates = vRowIndices[vL];

    if (vCandidates.empty()) continue;

    const float minU = uL - maxD;
    const float maxU = uL - minD;

    if (maxU < 0) continue;

    int bestDist = ORBmatcher::TH_HIGH;
    size_t bestIdxR = 0;

    const cv::Mat &dL = mDescriptors.row(iL);

    // Compare descriptor to right keypoints
    for (size_t iC = 0; iC < vCandidates.size(); iC++) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint &kpR = mvKeysRight[iR];

      if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) continue;

      const float &uR = kpR.pt.x;

      if (uR >= minU && uR <= maxU) {
        const cv::Mat &dR = mDescriptorsRight.row(iR);
        const int dist = ORBmatcher::DescriptorDistance(dL, dR);

        if (dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        }
      }
    }

    // Subpixel match by correlation
    if (bestDist < thOrbDist) {
      // coordinates in image pyramid at keypoint scale
      const float uR0 = mvKeysRight[bestIdxR].pt.x;
      const float scaleFactor = mvInvScaleFactors[kpL.octave];
      const float scaleduL = round(kpL.pt.x * scaleFactor);
      const float scaledvL = round(kpL.pt.y * scaleFactor);
      const float scaleduR0 = round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                       .rowRange(scaledvL - w, scaledvL + w + 1)
                       .colRange(scaleduL - w, scaleduL + w + 1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

      int bestDist = INT_MAX;
      int bestincR = 0;
      const int L = 5;
      vector<float> vDists;
      vDists.resize(2 * L + 1);

      const float iniu = scaleduR0 + L - w;
      const float endu = scaleduR0 + L + w + 1;
      if (iniu < 0 ||
          endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
        continue;

      for (int incR = -L; incR <= +L; incR++) {
        cv::Mat IR =
            mpORBextractorRight->mvImagePyramid[kpL.octave]
                .rowRange(scaledvL - w, scaledvL + w + 1)
                .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
        IR.convertTo(IR, CV_32F);
        IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

        float dist = cv::norm(IL, IR, cv::NORM_L1);
        if (dist < bestDist) {
          bestDist = dist;
          bestincR = incR;
        }

        vDists[L + incR] = dist;
      }

      if (bestincR == -L || bestincR == L) continue;

      // Sub-pixel match (Parabola fitting)
      const float dist1 = vDists[L + bestincR - 1];
      const float dist2 = vDists[L + bestincR];
      const float dist3 = vDists[L + bestincR + 1];

      const float deltaR =
          (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

      if (deltaR < -1 || deltaR > 1) continue;

      // Re-scaled coordinate
      float bestuR = mvScaleFactors[kpL.octave] *
                     ((float)scaleduR0 + (float)bestincR + deltaR);

      float disparity = (uL - bestuR);

      if (disparity >= minD && disparity < maxD) {
        if (disparity <= 0) {
          disparity = 0.01;
          bestuR = uL - 0.01;
        }
        mvDepth[iL] = mbf / disparity;
        mvuRight[iL] = bestuR;
        vDistIdx.push_back(pair<int, int>(bestDist, iL));
      }
    }
  }

  sort(vDistIdx.begin(), vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size() / 2].first;
  const float thDist = 1.5f * 1.4f * median;

  for (int i = vDistIdx.size() - 1; i >= 0; i--) {
    if (vDistIdx[i].first < thDist)
      break;
    else {
      mvuRight[vDistIdx[i].second] = -1;
      mvDepth[vDistIdx[i].second] = -1;
    }
  }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
  mvuRight = vector<float>(N, -1);
  mvDepth = vector<float>(N, -1);

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeys[i];
    const cv::KeyPoint &kpU = mvKeysUn[i];

    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = imDepth.at<float>(v, u);

    if (d > 0) {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - mbf / d;
    }
  }
}

cv::Mat Frame::UnprojectStereo(const int &i) {
  const float z = mvDepth[i];
  if (z > 0) {
    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << mvP3Dfisheye[i].x,mvP3Dfisheye[i].y,mvP3Dfisheye[i].z);
    return mRwc*x3Dc+mOw;
  } else
    return cv::Mat();
}


/*
 * @func: ComputeFisheyeStereoMatches()
 * @brief: 计算双目中左右目的匹配点
 * @note: 该函数主要的更新变量如下：
 *        mvuRight：左目特征点在右目中对应的横坐标x
 *        mvvRight：左目特征点在右目中对应的纵坐标y
 *        mvP3Dfisheye： 左目特征点对应的在世界坐标系下的三维坐标x,y,z
 *        mvDepth：左目特征点在对应的深度值，mvP3Dfisheye中的z值
 *
 * */
void Frame::ComputeFisheyeStereoMatches() {
  std::cout << "ComputeFisheyeStereoMatches " << std::endl;
  mvuRight = vector<float>(N, -1.0f);
  mvvRight = vector<float>(N, -1.0f);
  mvDepth = vector<float>(N, -1.0f);
  mvP3Dfisheye = vector<cv::Point3f>(N, cv::Point3f(-1, -1, -1));
  int matchedbyDes = 0;
  // 1. 初步搜索匹配点
  // 建立该vector用于保存右目中与当前index对应的特征点index
  vector<int> vnMatcheslr;
  vector<int> vbestdist;
  vbestdist = vector<int>(mvKeys.size(), -1);
  vnMatcheslr = vector<int>(mvKeys.size(), -1);
  const int Nl = mvKeys.size();
  for (int iL = 0; iL < Nl; iL++) {
    const cv::KeyPoint &kpl = mvKeys[iL];
    const float &kpY = kpl.pt.y;
    const float &kpX = kpl.pt.x;
    int levell = kpl.octave;
    int windowSize = 100;
    vector<size_t> vIndicesRight;
    // 目前只能在当前level匹配. 返回所有符合条件的特征点的index
    vIndicesRight =
        GetFeaturesInAreaRight(kpX, kpY, windowSize, levell, levell);
    if (vIndicesRight.empty()) continue;

    const cv::Mat &dL = mDescriptors.row(iL);
    int bestDist = INT_MAX;
    int bestDist2 = INT_MAX;
    int bestIdx2 = -1;
    for (vector<size_t>::iterator vit = vIndicesRight.begin();
         vit != vIndicesRight.end(); vit++) {
      size_t i2 = *vit;
      const cv::Mat &dR = mDescriptorsRight.row(i2);
      const int dist = ORBmatcher::DescriptorDistance(dL, dR);
      if (dist >= INT_MAX) continue;
      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = i2;
      } else if (dist < bestDist2)
        bestDist2 = dist;
    }

    //根据最佳距离筛选当前两个特征是否匹配
    if (bestDist <= ORBmatcher::TH_LOW) {
      if (bestDist < (float)bestDist2 * 0.6) {
        vnMatcheslr[iL] = bestIdx2;
        vbestdist[iL] = bestDist;
        matchedbyDes++;
      }
    }
  }

   std::cout << "matchedbyDes = " << matchedbyDes<<endl;
   int matchedbyTri = 0;
  // 2.从用匹配点进行三角化，将合格的点保存
  for (int iL = 0; iL < Nl; iL++) {
    if (vnMatcheslr[iL] < 0) {
      continue;
    }

    cv::Mat P3D1 = cv::Mat(3, 1, CV_32F);
    int iR = vnMatcheslr[iL];
    int ret = Check3Dloc(mvP3M, iL, mvP3MRight, iR, P3D1);
    if (ret == -1) {
      vnMatcheslr[iL] = -1;
      continue;
    }  //说明当前匹配对不合格
    matchedbyTri++;
    float stereo_depth = P3D1.at<float>(2);
    //if(stereo_depth >2.0) continue;   
    mvP3Dfisheye[iL] =
        cv::Point3f(P3D1.at<float>(0), P3D1.at<float>(1), P3D1.at<float>(2));
    mvDepth[iL] = P3D1.at<float>(2);
    // 用vnMatcheslr[iL]表示左目iL点在右目mvKeysRight中对应的index
    const cv::KeyPoint &kpr = mvKeysRight[vnMatcheslr[iL]];
    mvuRight[iL] = kpr.pt.x;
    mvvRight[iL] = kpr.pt.y;
  }

  std::cout << "matchedbyTri = " << matchedbyTri<< std::endl;
  for (int i = 0; i < vnMatcheslr.size(); i++) {
      mvMatcheslr.push_back(vnMatcheslr[i]);
  }  
}

int Frame::Check3Dloc(std::vector<cv::Point3f> left_P3M, int iL,
                      std::vector<cv::Point3f> right_P3M, int iR,
                      cv::Mat &p3dC1) {
  cv::Matx33d Rrl = mRrl.clone();
  cv::Matx31d tlinr;
  tlinr(0) = mtlinr.at<float>(0);
  tlinr(1) = mtlinr.at<float>(1);
  tlinr(2) = mtlinr.at<float>(2);

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
  K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

  cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
  P2.at<float>(0, 0) = Rrl(0, 0);
  P2.at<float>(0, 1) = Rrl(0, 1);
  P2.at<float>(0, 2) = Rrl(0, 2);
  P2.at<float>(1, 0) = Rrl(1, 0);
  P2.at<float>(1, 1) = Rrl(1, 1);
  P2.at<float>(1, 2) = Rrl(1, 2);
  P2.at<float>(2, 0) = Rrl(2, 0);
  P2.at<float>(2, 1) = Rrl(2, 1);
  P2.at<float>(2, 2) = Rrl(2, 2);
  P2.at<float>(0, 3) = tlinr(0);
  P2.at<float>(1, 3) = tlinr(1);
  P2.at<float>(2, 3) = tlinr(2);
  P2 = K * P2;
  cv::Matx31d trinl = -Rrl.t() * tlinr;
  cv::Mat O2 = cv::Mat::zeros(3, 1, CV_32F);
  O2.at<float>(0) = trinl(0);
  O2.at<float>(1) = trinl(1);
  O2.at<float>(2) = trinl(2);

  cv::Point3f p1m = left_P3M[iL];
  cv::Point3f p2m = right_P3M[iR];

  TriangulateFisheye(p1m, p2m, P1, P2, p3dC1);
  //计算该三维点在相机1和相机2中的重投影，如果误差较大，则忽略该点；
  //条件1：三维坐标点必须是有效值
  if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) ||
      !isfinite(p3dC1.at<float>(2)))
    return -1;

  // Check parallax
  cv::Mat normal1 = p3dC1 - O1;
  float dist1 = cv::norm(normal1);

  cv::Mat normal2 = p3dC1 - O2;
  float dist2 = cv::norm(normal2);

  float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

  if (cosParallax >= 0.99998) return -1;

  if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998) return -1;
  cv::Matx31d p3dC1_mat =
      cv::Matx31d(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
  cv::Matx31d p3dC2_v = Rrl * p3dC1_mat + tlinr;
  cv::Mat p3dC2 = cv::Mat::zeros(3, 1, CV_32F);
  p3dC2.at<float>(0) = p3dC2_v(0);
  p3dC2.at<float>(1) = p3dC2_v(1);
  p3dC2.at<float>(2) = p3dC2_v(2);

  // 条件3. 在相机2的非归一化相机坐标系中的坐标不能为负
  if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998) return -1;
  
  // cv::Point2f im1uv = mpCamera->World2Img(p3dC1);
  cv::Point2f im1uv;
  int ret1 = mpCamera->world2Img(p3dC1, im1uv);
  if (ret1 == -1)
    return -1;

  float imgx = im1uv.x;
  float imgy = im1uv.y;
  float squareError1 = (mvKeys[iL].pt.x - imgx) * (mvKeys[iL].pt.x - imgx) +
                       (mvKeys[iL].pt.y - imgy) * (mvKeys[iL].pt.y - imgy);
  float error_sigma = mvLevelSigma2[mvKeys[iL].octave];
  if (squareError1 > 5.991 * error_sigma)
    return -1;

  // Check reprojection error in second image
  cv::Point2f img2uv;
  int ret2 = mpCamera2->world2Img(p3dC2, img2uv);
  if (ret2 == -1)
    return -1;
  
  float squareError2 =
      (mvKeysRight[iR].pt.x - img2uv.x) * (mvKeysRight[iR].pt.x - img2uv.x) +
      (mvKeysRight[iR].pt.y - img2uv.y) * (mvKeysRight[iR].pt.y - img2uv.y);
  float error_sigma2 = mvLevelSigma2[mvKeysRight[iR].octave];
  if (squareError2 > 5.991 * error_sigma2)
    return -1;

  return 0;
}

/*
 * @func: 三角化恢复鱼眼镜头中特征点的三维距离
 * @param：p1m,p2m 特征点在相机坐标系下的坐标
 * @param: P1,P2
 * 投影矩阵，因为点的坐标已经转换到相机坐标系下，所以这里的投影矩阵与内参无关，只与Rt相关
 * */
void Frame::TriangulateFisheye(const cv::Point3f &p1m, const cv::Point3f &p2m,
                               const cv::Mat &P1, const cv::Mat &P2,
                               cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_32F);

  A.row(0) = p1m.x * P1.row(2) - p1m.z * P1.row(0);
  A.row(1) = p1m.y * P1.row(2) - p1m.z * P1.row(1);
  A.row(2) = p2m.x * P2.row(2) - p2m.z * P2.row(0);
  A.row(3) = p2m.y * P2.row(2) - p2m.z * P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

cv::Mat Frame::ProjectEUCM(float &camx, float &camy,cv::Mat &intrin_K){
  cv::Mat ptincam = cv::Mat::ones(3,1,CV_32F);
  ptincam.at<float>(0) = camx;
  ptincam.at<float>(1) = camy;
  cv::Mat ptinImg = cv::Mat::ones(3,1,CV_32F);
  ptinImg = intrin_K*ptincam;
  return ptinImg;
}

}  // namespace ORB_SLAM2
