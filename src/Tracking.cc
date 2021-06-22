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

#include "Tracking.h"

#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CameraModels/EUCM.h"
#include "CameraModels/KannalaBrandt8.h"

#include "Converter.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#define VISUAL
#define NOSCALE
using namespace std;

namespace ORB_SLAM2 {

Tracking::Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer,
                   MapDrawer* pMapDrawer, Map* pMap, KeyFrameDatabase* pKFDB,
                   const string& strSettingPath, const int sensor)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbOnlyTracking(false),
      mbVO(false),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpInitializer(static_cast<Initializer*>(NULL)),
      mpSystem(pSys),
      mpViewer(NULL),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpMap(pMap),
      mnLastRelocFrameId(0) {
  // Load camera parameters from settings file

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  ParseCamParamFile(fSettings);
  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30;
  mbf = fSettings["Camera.bf"];
  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if (mbRGB)
    cout << "- color order: RGB (ignored if grayscale)" << endl;
  else
    cout << "- color order: BGR (ignored if grayscale)" << endl;

  // Load ORB parameters

  if (sensor == System::RGBD) {
    mDepthMapFactor = fSettings["DepthMapFactor"];
    if (fabs(mDepthMapFactor) < 1e-5)
      mDepthMapFactor = 1;
    else
      mDepthMapFactor = 1.0f / mDepthMapFactor;
  }
  ParseORBParamFile(fSettings);
  // mrelocalnum = 0;

  mnMatches = fSettings["Track.nMatches"];
  mSearchWindowSize = fSettings["Track.SearchWindowSize"];
  mnFeaturesForStereoReinit = fSettings["Track.nFeaturesForStereoReinit"];
  mnThresTrackLocalMap = fSettings["Track.nThresTrackLocalMap"];

  mbDR = false;

  DR_x = DR_y = DR_th = DR_del_x = DR_del_y = DR_del_th = 0;

  IsScaled = false;
  scaleinTracking = false;  //initialized as false
  
  mTbc = cv::Mat::eye(4, 4, CV_32F);
  fSettings["Camera.Tbc"] >>  mTbc;
  cout << "CAMERA tBC: " << mTbc << endl;

  mRbc = mTbc.colRange(0,3).rowRange(0,3);
  mcam_in_baselink = mTbc.col(3).rowRange(0,3);
  mbaselink_in_cam = -mRbc.t()*mcam_in_baselink;
  
  mTcb = cv::Mat::eye(4,4,CV_32F);
  cv::Mat Rcb = mRbc.t();
  Rcb.copyTo(mTcb.colRange(0,3).rowRange(0,3));
  mbaselink_in_cam.copyTo(mTcb.col(3).rowRange(0,3));

  mDR_Tcw = cv::Mat::eye(4, 4, CV_32F);
  mTcw_1 = cv::Mat::eye(4, 4, CV_32F);

  // scale estimator parameters
  float minDistDR = fSettings["ScaleEst.minDistDR"];
  int minKFs = fSettings["ScaleEst.mMinKFs"];
  float scaleDriftThresh = fSettings["ScaleEst.mScaleDriftThresh"];
  float minDist = fSettings["ScaleEst.mMinDist"];
  float deLimit = fSettings["ScaleEst.mDELimit"];

  mScaleEst = ScaleEstimator(minDistDR,minKFs,scaleDriftThresh,minDist,deLimit,mSensor);
  numofRelocFrame=0;
  mbWithMap = false;
  OdomExist = false;
  std::cout << "----------------FInished------" << std::endl;
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat DistCoef(4, 1, CV_32F);
  float fx,fy,cx,cy,k1,k2,k3,k4;
  string sCameraName = fSettings["Camera.type"];
  std::cout << "sCameraName: " << sCameraName << std::endl;
  if(sCameraName == "EUCM"){
    fx = fSettings["Camera.fx"];
    fy = fSettings["Camera.fy"];
    cx = fSettings["Camera.cx"];
    cy = fSettings["Camera.cy"];
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);
    std::cout << "[EUCM] K: " << endl << K <<endl;
    k1 = fSettings["Camera.k1"];
    k2 = fSettings["Camera.k2"];
    k3 = fSettings["Camera.k3"];
    k4 = fSettings["Camera.k4"];
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = k3;
    DistCoef.at<float>(3) = k4;

    DistCoef.copyTo(mDistCoef);
    std::cout << "[EUCM] DistCoef: " << endl << mDistCoef <<endl;
    vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
    mpCamera = new EUCM(vCamCalib);

    if (mSensor == System::STEREO) {
      //Init Second Camera
      cv::Mat tmp_K_r, tmp_DistCoef_r;
      fSettings["RIGHT.K"] >> tmp_K_r;
      fSettings["RIGHT.D"] >> tmp_DistCoef_r;
      cv::Mat K_r = cv::Mat::eye(3, 3, CV_32F);
      tmp_K_r.convertTo(K_r, CV_32F);
      cv::Mat DistCoef_r(5,1,CV_32F);
      tmp_DistCoef_r.convertTo(DistCoef_r,CV_32F);
      float fx_r,fy_r,cx_r,cy_r,k1_r,k2_r,k3_r,k4_r;
      fx_r = K_r.at<float>(0,0);
      fy_r = K_r.at<float>(1,1);
      cx_r = K_r.at<float>(0,2);
      cy_r = K_r.at<float>(1,2);
      k1_r = DistCoef_r.at<float>(0);
      k2_r = DistCoef_r.at<float>(1);
      k3_r = DistCoef_r.at<float>(2);
      k4_r = DistCoef_r.at<float>(3);

      vector<float> vCamCalib2{fx_r,fy_r,cx_r,cy_r,k1_r,k2_r,k3_r,k4_r};
      mpCamera2 = new EUCM(vCamCalib2);
      std::cout << "[EUCM] K_r: " << endl << K_r <<endl;
      std::cout << "[EUCM] DistCoef_r: " << endl << DistCoef_r <<endl;
      cv::Mat tmp_RotationRL, tmp_tlinr;
      fSettings["CAMERA.R"] >> tmp_RotationRL;
      fSettings["CAMERA.T"] >> tmp_tlinr;
      cv::Mat RotationRL(3, 3, CV_32F);
      tmp_RotationRL.convertTo(RotationRL, CV_32F);
      cv::Mat tlinr(3, 1, CV_32F);
      tmp_tlinr.convertTo(tlinr, CV_32F);
  
      RotationRL.copyTo(mRrl);
      tlinr.copyTo(mtlinr);
    }
  }
  else if(sCameraName == "KB")
  {
    fx = fSettings["Camera.fx"];
    fy = fSettings["Camera.fy"];
    cx = fSettings["Camera.cx"];
    cy = fSettings["Camera.cy"];
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);
    std::cout << "[KannalaBrandt8] K: " << endl << K <<endl;
    k1 = fSettings["Camera.k1"];
    k2 = fSettings["Camera.k2"];
    k3 = fSettings["Camera.p1"];
    k4 = fSettings["Camera.p2"];
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = k3;
    DistCoef.at<float>(3) = k4;

    DistCoef.copyTo(mDistCoef);
    std::cout << "[KannalaBrandt8] mDistCoef: " << endl << mDistCoef <<endl;
    vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
    mpCamera = new KannalaBrandt8(vCamCalib);

    if (mSensor == System::STEREO) {
      // Init Second Camera
      cv::Mat tmp_K_r, tmp_DistCoef_r;
      fSettings["RIGHT.K"] >> tmp_K_r;
      fSettings["RIGHT.D"] >> tmp_DistCoef_r;
      cv::Mat K_r = cv::Mat::eye(3, 3, CV_32F);
      tmp_K_r.convertTo(K_r, CV_32F);

      cv::Mat DistCoef_r(5, 1, CV_32F);
      tmp_DistCoef_r.convertTo(DistCoef_r, CV_32F);
      float fx_r, fy_r, cx_r, cy_r, k1_r, k2_r, k3_r, k4_r;
      fx_r = K_r.at<float>(0, 0);
      fy_r = K_r.at<float>(1, 1);
      cx_r = K_r.at<float>(0, 2);
      cy_r = K_r.at<float>(1, 2);
      k1_r = DistCoef_r.at<float>(0);
      k2_r = DistCoef_r.at<float>(1);
      k3_r = DistCoef_r.at<float>(2);
      k4_r = DistCoef_r.at<float>(3);

      vector<float> vCamCalib2{fx_r,fy_r,cx_r,cy_r,k1_r,k2_r,k3_r,k4_r};
      mpCamera2 = new KannalaBrandt8(vCamCalib2);

      std::cout << "[KannalaBrandt8] K_r: " << endl << K_r <<endl;
      std::cout << "[KannalaBrandt8] DistCoef_r: " << endl << DistCoef_r <<endl;
      
      cv::Mat tmp_RotationRL, tmp_tlinr;
      fSettings["CAMERA.R"] >> tmp_RotationRL;
      fSettings["CAMERA.T"] >> tmp_tlinr;
      cv::Mat RotationRL(3, 3, CV_32F);
      tmp_RotationRL.convertTo(RotationRL, CV_32F);
      cv::Mat tlinr(3, 1, CV_32F);
      tmp_tlinr.convertTo(tlinr, CV_32F);
  
      RotationRL.copyTo(mRrl);
      tlinr.copyTo(mtlinr);
    }
  }else{
    cout << "!!!!!CAUTION: Unknown Camera Model!!!" << endl;
    return -1;
  }



  if (mSensor == System::STEREO || mSensor == System::RGBD) {
    mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
  }
  return 1;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
  int nFeatures = fSettings["ORBextractor.nFeatures"];
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  int nLevels = fSettings["ORBextractor.nLevels"];
  int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
  int fMinThFAST = fSettings["ORBextractor.minThFAST"];

  mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                        fIniThFAST, fMinThFAST);

  if (mSensor == System::STEREO)
    mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                           fIniThFAST, fMinThFAST);

  if (mSensor == System::MONOCULAR)
    mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  cout << endl << "ORB Extractor Parameters: " << endl;
  cout << "- Number of Features: " << nFeatures << endl;
  cout << "- Scale Levels: " << nLevels << endl;
  cout << "- Scale Factor: " << fScaleFactor << endl;
  cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
  cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
  return 1;
}

void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer* pViewer) { mpViewer = pViewer; }

cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  const double& timestamp) {
  std::cout << "GrabImageStereo()" << std::endl;
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  mCurrentFrame =
      Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft,
            mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mRrl, mtlinr, mbf,
            mThDepth, mpCamera, mpCamera2, DR_x, DR_y, OdomExist, mSensor, mTbc);
  cout << " ------------ Frame ID: " << mCurrentFrame.mnId << " -------------"
       << endl;
  Track();
  cout << " ------------ mState: " << mState << " -----------------" << endl;
  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                const double& timestamp) {
  mImGray = imRGB;
  cv::Mat imDepth = imD;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

  mCurrentFrame =
      Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary,
            mK, mDistCoef, mbf, mThDepth, mSensor, mTbc);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im,
                                     const double& timestamp) {
  mImGray = im;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET ||
      mbDR == true)
  {
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, DR_x,
                          DR_y, OdomExist, mSensor, mTbc);
  }
  else
  {
    mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, DR_x,
                          DR_y, OdomExist, mSensor, mTbc);
  }

  cout << " ------------ Frame ID: " << mCurrentFrame.mnId << " -------------"
       << endl;

  Track();
  cout << " ------------ mState: " << mState << " -----------------" << endl;
  return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }
  mLastProcessedState = mState;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
  if (mState == NOT_INITIALIZED) 
  {
    if (mpMap->GetMaxKFid() > 0 && numofRelocFrame < 60) //monocular initializated with map
    {
      MonocularInitializationwithMap();
      numofRelocFrame++;
      mbWithMap = true;
      mpFrameDrawer->Update(this);
      std::cout << "mState = " << mState << "and numofRelocFrame = " << numofRelocFrame << std::endl;
      if(mState!=OK){
          cout << "mState !=OK, return ." << std::endl;
          mState = NOT_INITIALIZED;
          return;            
      }
      if(mState == OK){
          cout << "mState is OK after MonoInitnwithMap()"<<endl;
          cout << "current pose id is : "<< mCurrentFrame.mnId << "with pose "<<endl <<mCurrentFrame.mTcw<<endl;
          numofRelocFrame = 0; //shinan: 初始化成功后设置为0
          mLastFrame = Frame(mCurrentFrame); //shinan: 没用relocal成功就不需要mLastFram吧
          mScaleEst.ScaleEstreset(DR_x,DR_y,mCurrentFrame.GetRobotCenter());
          mScaleEst.setFirstKFId(mpReferenceKF->mnId); //NOTE: temp set but actually mFirstKFId is only used for debug
          IsScaled = false;
      }
    }
    else if(mpMap->GetMaxKFid() == 0 || numofRelocFrame >=60)
    {
        if(mState!=OK)
        {
          if (mCurrentFrame.mnId == 0){
              SetFirstFramePose(mCurrentFrame.mDrX, mCurrentFrame.mDrY, DR_th);
          }else{
              TrackWithDR();
          }
        }
        std::cout << "mState = NOT_INITIALIZED"<< endl;
        if (mSensor == System::STEREO || mSensor == System::RGBD)
          StereoInitialization();//双目初始化的位姿设定为mDR_Tcw
        else {
          MonocularInitializationWithEncoder();
          mInitialDRForResize.x = DR_x;
          mInitialDRForResize.y = DR_y;
        }

  
        if (mState != OK) {
          mCurrentFrame.mTcw = mTcw_1.clone();
          #ifdef VISUAL
          mpFrameDrawer->Update(this);
          #endif
          mCurrentFrame.mpReferenceKF = mpReferenceKF;
          mLastFrame = Frame(mCurrentFrame);
          return;
        }else{
            #ifdef VISUAL
            mpFrameDrawer->Update(this);
            #endif
        }
        if (mbDR) mbDR = false;  // temp!!!
    }
  } 
  else {// System is initialized. Track Frame.
    bool bOK;
    if (!mbOnlyTracking) 
    {
      if (mState == OK) 
      {
          CheckReplacedInLastFrame();
          mTcw_1 = mLastFrame.mTcw.clone();
          TrackWithDR();
          bOK = TrackWithTFPose();
          if(!bOK){
            std::cout << "!!!!!!!!!!!!!!!!!!Failed track with TF POSE " << std::endl;
            if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 5) {
              bOK = TrackReferenceKeyFrame();
            } else {
              bOK = TrackWithMotionModel();
              if (!bOK) bOK = TrackReferenceKeyFrame();
            }
          }
      } 
      else //LOST
      {
        TrackWithDR();
        std::cout << "mState == LOST" << endl;
        if(mbWithMap && numofRelocFrame <60 ){
            cout << " [Track] ---Relocalization " <<endl; 
            bOK = Relocalization();
            numofRelocFrame++;
        }
        else if (mbDR==true || numofRelocFrame >=60 ) //shinan added for RelocWithMap 连续10帧都没有重定位成功，则进入DR模式重新初始化 
        {
          numofRelocFrame = 0;
          if(mSensor==System::STEREO || mSensor==System::RGBD)
              StereoReinitialization();    
          else
              //MonocularInitialization();  // temp!!!
              MonocularReInitializationWithEncoder();
          if (mState==OK)
          {   
              mpMap->ResetKFCounter();       
              if(mSensor==System::STEREO || mSensor==System::RGBD){}
              else
                  cv::Mat Tcw = mCurrentFrame.mTcw.clone();
              mbDR = false;
          }

          if(mState!=OK ) { // DR
              if (mState==LOST && mbDR){
                  cout << "Lost n Renitialized Failed" << endl;
                  mCurrentFrame.mTcw = mTcw_1.clone();
              }   // DR
              //如果mstate不等于OK,则为SLAM创建丢失关键帧 如果mState=OK,则不再需要创建DR关键帧
              if(NeedNewKeyFrameDR()) CreateNewKeyFrameDR();
          }
          #ifdef VISUAL
          mpFrameDrawer->Update(this);
          #endif
          mCurrentFrame.mpReferenceKF = mpReferenceKF; 
          mLastFrame = Frame(mCurrentFrame);
          // Store frame pose information to retrieve the complete camera trajectory afterwards.
          if(!mCurrentFrame.mTcw.empty())
          {
              cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
              mCurrentFrame.RefFramePoseTcw = mCurrentFrame.mpReferenceKF->GetPose();
              mlRelativeFramePoses.push_back(Tcr);
              mlpReferences.push_back(mpReferenceKF);
              mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
              mlbLost.push_back(mState==LOST);
          }
          else
          {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
          }
          return;
        } 
      }
    }
    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    if (!mbOnlyTracking) {
      if (bOK) bOK = TrackLocalMap();
    } 


    if (bOK)
      mState = OK;
    else
      mState = LOST;
    
    
    if (mbWithMap && mState ==OK &&  numofRelocFrame>0){
        numofRelocFrame = 0;
        mbDR=false;
    }

    // 这一步的执行还是在currentFrame, 还没有进行到下一帧图像, 也就是在currentFrame丢失之后,尽力挽救一下
    // DR模式未开启时才需要这一步，如果已经开启后重复执行这一步，会覆盖mTcw_1
    if (mState == LOST && !mbDR) {
      {
        #ifndef NOSCALE
        ScaleforLost(); 
        #endif
      }
      mTcw_1 = mLastFrame.mTcw.clone();
      TrackWithDR();
      //利用TrackWithDR()的结果再进行一次的Track尝试；
      {
          bool reTrack_OK = ReTrackWithMotionModel(mDR_Tcw);
          if (reTrack_OK)
            reTrack_OK = TrackLocalMap();

          if(reTrack_OK){
              mState =OK;  //不用跳去哪里，直接顺序执行，接下来updater drawer就行；因为正常如果不Lost的话，也会执行到那里；
              bOK = true;
              std::cout << "YEAH!!!  Tracking success after ReTrack!!!!!" << endl;
          }
          else
          {
              mState = LOST;
              mbDR = true;
              std::cout << "mState = LOST and DR is true " << std::endl;
              mCurrentFrame.mTcw = mTcw_1.clone();
              cout << "set mTcw_1  as currentFrame pose" << mTcw_1 <<endl;
              mVelocity = cv::Mat();
          }    
      }
    }

    // Update drawer
    #ifdef VISUAL
    mpFrameDrawer->Update(this);
    #endif
    // If tracking were good, check if we insert a keyframe
    if (bOK) 
    {
      if (!mLastFrame.mTcw.empty())
      { // Update motion model
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo( LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
      } else
        mVelocity = cv::Mat();
      #ifdef VISUAL
      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
      #endif
      // Clean VO matches
      for (int i = 0; i < mCurrentFrame.N; i++)
      {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),lend = mlpTemporalPoints.end();lit != lend; lit++) {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) CreateNewKeyFrame();

      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        cout << "Track lost soon after initialisation, reseting..." << endl;
        mpSystem->Reset();
        return;
        }else if(NeedNewKeyFrameDR()) CreateNewKeyFrameDR();
    }

    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    if (!mLastFrame.mTcw.empty()) mLastTcw = mLastFrame.mTcw.clone();
    mLastFrame = Frame(mCurrentFrame);
  }

  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mCurrentFrame.RefFramePoseTcw = mCurrentFrame.mpReferenceKF->GetPose();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
  cout << "[Track] Returning Track() with Tcw = " << mCurrentFrame.mTcw << endl;
}

void Tracking::ScaleforLost()
{
  //cout << "[TR] " << __FUNCTION__<< endl;
  {
    mScaleEst.computeScale();
    double lostscale = mScaleEst.getScale();
    //cout << "    lostscale is " << lostscale << endl;
    if(lostscale > 0.000001 ){
        ResizeMapData(lostscale);  
        // Resize Last Frame for better calcu for mTcw_1
        cv::Mat lastResizepose = mLastResizePose.clone();
        ResizePose(lastResizepose, mLastFrame,lostscale);     
    }

    mScaleEst.resetParams();
  } //shinan: no need to set the last resize frame, cuz renitialization will do .

}

void Tracking::StereoInitialization() {
  if (mCurrentFrame.N > 500) {
    std::cout << __FUNCTION__ << endl;
    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

    if(!mDR_Tcw.empty()){
     mCurrentFrame.SetPose(mDR_Tcw);
    }

    // Create KeyFrame
    KeyFrame *pKFini =
        new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB, mSensor, mTbc);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
      }
    }

    cout << "New map created with " << mpMap->MapPointsInMap() << " points"
         << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    pKFini->mbInitKF = true;
#ifdef VISUAL
    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
#endif
    mState = OK;
    mLastResizeFrameID = mCurrentFrame.mnId;
    mLastResizePose = mCurrentFrame.mTcw;
  }
}

void Tracking::MonocularInitializationwithMap()
{
    cout<< "------------------RELOCAL WITH MAP------------------" <<std::endl;
    bool bRelocOk = false;
    if(mCurrentFrame.N < 500) return;
    {
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        bRelocOk = Relocalization(); 
        cout << "After relocal mCurrentFrame set to " << mCurrentFrame.mTcw <<  endl;    
        //mCurrentFrame.mpReferenceKF = mpReferenceKF; //shinannoted: 觉得没用，这一步会在TrackLOcalMap里首先进行
    }
    if(!mbOnlyTracking)
    {
        if(bRelocOk)
            bRelocOk = TrackLocalMap();
    }else
    {
        if(bRelocOk && !mbVO)
            bRelocOk = TrackLocalMap();
    }

    //shinan: temp add to check
    if(bRelocOk)
    {
        cout << "mpReferenceKF set as " <<mpReferenceKF->mnId<<endl;
    }

    if(bRelocOk)
        mState = OK;
    else
        mState = LOST;

    if(bRelocOk)
    {
        // Update motion model
        if(!mLastFrame.mTcw.empty())
        {
            //std::cout << "mLastFrame is " << mLastFrame.mnId << " with pose " <<  mLastFrame.mTcw <<endl;
            cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
            mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
            mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
            mVelocity = mCurrentFrame.mTcw*LastTwc;
            //std::cout << " set velocity " <<mVelocity << endl;
        }
        else
            mVelocity = cv::Mat();
        #ifdef VISUAL
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        #endif
        // Clean VO matches
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
                if(pMP->Observations()<1)
                {
                    mCurrentFrame.mvbOutlier[i] = false;
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }
        }
        // Delete temporal MapPoints
        for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            delete pMP;
        }
        mlpTemporalPoints.clear();
        // Check if we need to insert a new keyframe
        if(NeedNewKeyFrame())
            CreateNewKeyFrame();
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
        }
    }

    std::cout << "bRelocOk is " << bRelocOk << std::endl;
    // Reset if the camera get lost soon after initialization
    if(mState==LOST)
    {
        if(mpMap->KeyFramesInMap()<=5)
        {
            cout << "  Track lost soon after initialisation, reseting..." << endl;
            mpSystem->Reset();
            return;
        }
    }

    if(!mCurrentFrame.mpReferenceKF && mpReferenceKF!=nullptr )
    {
        //cout << "   set mCurrentFrame.ReferenceKF and id is "<< mpReferenceKF->mnId <<endl;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
    if (!mLastFrame.mTcw.empty()){
        //cout << " set mLastTcw cuz it is not empty with id: " <<mLastFrame.mnId  <<endl;                
        mLastTcw = mLastFrame.mTcw.clone();
    }
    //cout << "     Ready to set mCurrentFrame as mLastFrame " << endl;
}

void Tracking::MonocularInitialization() {
  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer) delete mpInitializer;

      // mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
      float mfxSigma = mK.at<float>(0, 0);
      mpInitializer = new Initializer(mCurrentFrame, mfxSigma, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      return;
    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 70) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      //cout << "MonocularInitialization(): not enough maches" << endl;
      return;
    }

    cv::Mat Rcw;                  // Current Camera Rotation
    cv::Mat tcw;                  // Current Camera Translation
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                  mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::MonocularInitializationWithEncoder() {
  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100 && mCurrentFrame.mOdomFlag) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer) delete mpInitializer;

      float mfxSigma = mK.at<float>(0, 0);
      mpInitializer = new Initializer(mCurrentFrame, mfxSigma, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      mInitialDRForInit.x = DR_x;
      mInitialDRForInit.y = DR_y;

      mInitDR_Tcw = mDR_Tcw;

      mTcw0 = cv::Mat::zeros(4, 4, CV_32F);

      mLastTcwForInit = cv::Mat::zeros(4, 4, CV_32F);

      //cout << "mInitialDR update" << endl;
      return;
    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= 100 || (!mCurrentFrame.mOdomFlag)) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }
    //cout << "Try to initialize " << endl;
    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 70) {
      //cout <<"[MonoInitialization] nametches is " << nmatches <<" < 50 " << endl;
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      return;
    }

    cv::Mat Rcw;                  // Current Camera Rotation
    cv::Mat tcw;                  // Current Camera Translation
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                  mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      mCurrentDR.x = DR_x;
      mCurrentDR.y = DR_y;

      if (!mbDR) {
        // CreateInitialMapMonocular();
        CreateInitialMapMonocularWithEncoder();
      } else {
        ReCreateInitialMapMonocularWithEncoder();
      }

      mLastResizeFrameID = mCurrentFrame.mnId;
      mLastResizePose = mCurrentFrame.mTcw;
    }
  }
}

void Tracking::MonocularReInitializationWithEncoder() {
  //cout << "  MonocularReinitialization()" << endl;

  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100 && mCurrentFrame.mOdomFlag) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer) delete mpInitializer;

      float mfxSigma = mK.at<float>(0, 0);
      mpInitializer = new Initializer(mCurrentFrame, mfxSigma, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      mInitialDRForInit.x = DR_x;
      mInitialDRForInit.y = DR_y;

      mInitDR_Tcw = mDR_Tcw;

      mTcw0 = cv::Mat::zeros(4, 4, CV_32F);
      mLastTcwForInit = cv::Mat::zeros(4, 4, CV_32F);

      //cout<<"mReInitialDR update"<<endl;
      return;

    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= 100 || !mCurrentFrame.mOdomFlag) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // distance between 1st DR and current DR
    double dist_dr =
        sqrt((mInitialDRForInit.x - DR_x) * (mInitialDRForInit.x - DR_x) +
             (mInitialDRForInit.y - DR_y) * (mInitialDRForInit.y - DR_y));
    if (dist_dr < 0.001) {
      cout << "dist_dr = " << dist_dr ;
      cout << "  not enough motion for initialization" << endl;
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }
    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 70) {
      cout <<"[MonoReInitialization] nametches is " << nmatches <<" < 50 " << endl;
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      return;
    }

    cv::Mat Rcw;                  // Current Camera Rotation
    cv::Mat tcw;                  // Current Camera Translation
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                  mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      // set initial frame and current frame using initial guess
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      mCurrentDR.x = DR_x;
      mCurrentDR.y = DR_y;

      // then run BA to refine pose of frames and key points and create a map
      ReCreateInitialMapMonocularWithEncoder();

      scaleinTracking = false;  // set this to false incase for sudden scaling
    }
  }
}

void Tracking::StereoReinitialization() {
  cout << "  StereoReinitialization()" << endl;

  if (mCurrentFrame.N > mnFeaturesForStereoReinit) {
    mCurrentFrame.SetPose(mDR_Tcw);
    // Create KeyFrame
    KeyFrame *pKFini =
        new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB, mSensor, mTbc);

    pKFini->AddDRKeyFrame(mpLastKeyFrame);
    pKFini->mbDRKF = false;
    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);
    int ptnumber = 0;
    // Create MapPoints and asscoiate to KeyFrame
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
        ptnumber++;
      }
    }
    pKFini->UpdateConnections();
    cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;
    cout << "ReCreated map with " << ptnumber<< " mappoints " << endl;
    mpLocalMapper->InsertKeyFrame(pKFini);
    pKFini->mbInitKF = true;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    #ifdef VISUAL
    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
    #endif

    mState = OK;
    if(mbDR) mbDR=false;

    mLastResizeFrameID = mCurrentFrame.mnId;
    mLastResizePose = mCurrentFrame.mTcw;

    mLastFrame = Frame(mCurrentFrame);
    mpLastKeyFrame = pKFini;
    mnLastKeyFrameId = mCurrentFrame.mnId;
  }
}

void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame* pKFini =
      new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB, mSensor,mTbc);
  KeyFrame* pKFcur =
      new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB, mSensor,mTbc);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0) continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << mpMap->MapPointsInMap() << " points"
       << endl;

  Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 70) {
    cout << "Wrong initialization, reseting..." << endl;
    Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
  #ifdef VISUAL
  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
  #endif
  mpMap->mvpKeyFrameOrigins.push_back(pKFini);
  pKFini->mbInitKF = true;
  mState = OK;
}

void Tracking::CreateInitialMapMonocularWithEncoder() {
  cout << "[TR] CreateInitialMapMonocularWithEncoder()" << endl;

  // Create KeyFrames
  KeyFrame* pKFini =
      new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB,  mSensor,mTbc);
  KeyFrame* pKFcur =
      new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,  mSensor,mTbc);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0) continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    mpMap->AddMapPoint(pMP);
  }

  pKFini->mbInitKF = true;
  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << mpMap->MapPointsInMap() << " points"
       << endl;

  Optimizer::GlobalBundleAdjustemnt(mpMap, 20, false);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  double dist1 = sqrt((mInitialDRForInit.x - mCurrentDR.x) *
                          (mInitialDRForInit.x - mCurrentDR.x) +
                      (mInitialDRForInit.y - mCurrentDR.y) *
                          (mInitialDRForInit.y - mCurrentDR.y));
  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) {
    cout << "Wrong initialization, reseting..." << endl;
    if (mbDR == false)  // temp!!!
      Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  double dist2 = sqrt(Tc2w.at<float>(0, 3) * Tc2w.at<float>(0, 3) +
                      Tc2w.at<float>(1, 3) * Tc2w.at<float>(1, 3) +
                      Tc2w.at<float>(2, 3) * Tc2w.at<float>(2, 3));
  cv::Mat t = pKFcur->GetCameraCenter();
  cv::Mat Rwc(3, 3, CV_32F);
  cv::Mat twc(3, 1, CV_32F);
  Rwc = Tc2w.rowRange(0, 3).colRange(0, 3).t();
  twc = -Rwc * Tc2w.rowRange(0, 3).col(3);
  #ifndef NOSCALE
  invMedianDepth = dist1 / dist2;
  #endif
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;

  cout << "pKFini->mnId = " << pKFini->mnId << endl;
  cout << "pKFcur->mnId = " << pKFcur->mnId << endl;
  
  pKFini->SetScaleFlag();
  pKFcur->SetScaleFlag();
  
  pKFini->SetPose(mInitDR_Tcw);
  pKFcur->SetPose(Tc2w * mInitDR_Tcw);
  
  // Scale points
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      cv::Mat pos = pMP->GetWorldPos() * invMedianDepth;
      cv::Mat posH = cv::Mat::ones(4,1,CV_32F);
      pos.copyTo(posH.rowRange(0,3));
      posH = mInitDR_Tcw.inv() * posH;
      pos.at<float>(0) = posH.at<float>(0) / posH.at<float>(3);
      pos.at<float>(1) = posH.at<float>(1) / posH.at<float>(3);
      pos.at<float>(2) = posH.at<float>(2) / posH.at<float>(3);
      pMP->SetWorldPos(pos);
      pMP->SetScaleFlag();
    }
  }

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mScaleEst.ScaleEstreset(DR_x,DR_y,mCurrentFrame.GetRobotCenter());
  mScaleEst.setFirstKFId(pKFini->mnId);

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
#ifdef VISUAL
  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
#endif
  mpMap->mvpKeyFrameOrigins.push_back(pKFini);
  mState = OK;
}

/**
 * @brief create initial map after tracking fail
 *
 * similiar to the first initialization CreateInitialMapMonocularWithEncoder()
 * make a temp Map with initial KFs located on origin(0,0,0) for BA
 * and then move the KFs to the locations according to DR location after BA
 *
 */
void Tracking::ReCreateInitialMapMonocularWithEncoder() {
  cout << "[Tracking] ReCreateInitialMapMonocularWithEncoder()" << endl;
  Map* pMap = new Map();  // temp map for BA

  // Create KeyFrames
  KeyFrame* pKFini =
      new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB,  mSensor,mTbc);
  KeyFrame* pKFcur =
      new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,  mSensor,mTbc);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  pMap->AddKeyFrame(pKFini);
  pMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
  for(size_t i=0; i<mvIniMatches.size();i++)
  {
      if(mvIniMatches[i]<0)
          continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    pMap->AddMapPoint(pMP);
  }

  Optimizer::GlobalBundleAdjustemnt(pMap, 20, false);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  double dist1 = sqrt((mInitialDRForInit.x - mCurrentDR.x) *
                          (mInitialDRForInit.x - mCurrentDR.x) +
                      (mInitialDRForInit.y - mCurrentDR.y) *
                          (mInitialDRForInit.y - mCurrentDR.y));

  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) {
    cout << "Wrong initialization, reseting..." << endl;

    if (mbDR == false)  // temp!!!
      Reset();
    else{
      if(mpInitializer)
      {
          delete mpInitializer;
          mpInitializer = static_cast<Initializer*>(NULL);
      }
    }
    return;
  }

  // Scale initial baseline
  cv::Mat Twc1 = pKFini->GetPoseInverse();
  cv::Mat Tc2w = pKFcur->GetPose();
  cv::Mat Tc2c1 = Tc2w*Twc1;

  double dist2 = sqrt(Tc2c1.at<float>(0, 3) * Tc2c1.at<float>(0, 3) +
                      Tc2c1.at<float>(1, 3) * Tc2c1.at<float>(1, 3) +
                      Tc2c1.at<float>(2, 3) * Tc2c1.at<float>(2, 3));

  float scaleFactor = dist1 / dist2;
  Tc2c1.col(3).rowRange(0, 3) = Tc2c1.col(3).rowRange(0, 3) * scaleFactor;

  cout << "pKFini->mnId = " << pKFini->mnId << endl;
  cout << "pKFcur->mnId = " << pKFcur->mnId << endl;

  // transform the initial frames(pKFini, pKFcur) accoring to the last DR result
  pKFini->SetPose(mInitDR_Tcw);
  pKFcur->SetPose(Tc2c1 * mInitDR_Tcw);

  pKFini->AddDRKeyFrame(mpLastKeyFrame);  // for DR
  pKFcur->AddDRKeyFrame(pKFini);

  // Scale points
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];

      cv::Mat pos = pMP->GetWorldPos() * scaleFactor;
      cv::Mat posH = cv::Mat::ones(4,1,CV_32F);
      pos.copyTo(posH.rowRange(0,3));
      posH = mInitDR_Tcw.inv() * posH;
      pos.at<float>(0) = posH.at<float>(0) / posH.at<float>(3);
      pos.at<float>(1) = posH.at<float>(1) / posH.at<float>(3);
      pos.at<float>(2) = posH.at<float>(2) / posH.at<float>(3);

      pMP->SetWorldPos(pos);
      pMP->SetScaleFlag();
    }
  }
  
  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;
  
  pKFini->SetScaleFlag();
  pKFcur->SetScaleFlag();
  pKFini->mbInitKF = true;
    // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  mScaleEst.ScaleEstreset(DR_x, DR_y, mCurrentFrame.GetRobotCenter()); 
  mScaleEst.setFirstKFId(pKFini->mnId);

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
#ifdef VISUAL
  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
#endif
  mpMap->mvpKeyFrameOrigins.push_back(pKFini);
  
  mState = OK;
  // 重新初始化成功后则退出mbDR模式
  if(mbDR) mbDR==false;
  IsScaled = false;

  mnLastRelocFrameId = mCurrentFrame.mnId;
  
  mLastResizeFrameID = mCurrentFrame.mnId;
  mLastResizePose = pKFcur->GetPose().clone();
  mScaleEst.mLastResizeFrameID = mLastResizeFrameID;
  cout << "Finished Reinitialization and set mnLastResizeFrameId as " <<mLastResizeFrameID << std::endl;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame() {
  // cout << "[Tracking] Track reference frame  with refkf id " << mpReferenceKF->mnId << endl;
  // Compute Bag of Words vector
  mCurrentFrame.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.7, true);
  vector<MapPoint*> vpMapPointMatches;

  int nmatches =
      matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  // cout << "nmatches: " << nmatches << endl;

  if (nmatches < 15) return false;
  std::cout << __FUNCTION__ << " mLastFrame ID = " << mLastFrame.mnId << ", mpReferenceKF = " << 
  mpReferenceKF->mnFrameId << std::endl;
  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);//TODO: 为什么这里要用mLastFrame的pose?

  Optimizer::PoseOptimization(&mCurrentFrame);

  // cout << "[Tracking] pose after poseoptimization " << mCurrentFrame.mTcw <<  endl;
// scale estimation
//#ifdef SCALE_ESTIMATION
#if 0
    // dist between mLastFrame and mCurrentFrame
    // dist between DR_x,DR_y = DR_del_x, DR_del_y
    float dist_vo = DistBetweenFrames(mLastFrame, mCurrentFrame);
    float dist_dr = sqrt( DR_del_x*DR_del_x + DR_del_y*DR_del_y);
    cout<<"dist_vo="<<dist_vo<<", dist_dr="<<dist_dr<<endl; 
    float scale = 0.0;
    if (dist_vo==0 || dist_dr==0) {
        scale = 1.0;
    } else 
        scale = dist_dr / dist_vo;

    
    //scale = 0.4;
    cout<<"scale="<<scale<<endl;    
    // resize pose
    ResizePose(mLastFrame, mCurrentFrame, scale);

#endif

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        //TODO: 添加判断设置pMP->mbTrackInViewR的值
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
//#ifdef SCALE_ESTIMATION
#if 0
                // resize point
                ResizePoint(mLastFrame, pMP, scale);
#endif
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  // cout << "nmatchesMap: " << nmatchesMap << endl;
  return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();
   // mLastFrame.SetPose(Tlr*pRef->GetPose()); // no need to resize lastFrame because Ref has been resized
    
    if(!mLastFrame.RefFramePoseTcw.empty()){
        mLastFrame.SetPose(Tlr*mLastFrame.RefFramePoseTcw);
    }
    #ifdef NOSCALE
    scaleinTracking = false;
    #endif
    // Update pose according to reference keyframe
    if(scaleinTracking)
    {
        ResizeMapData(detectedscale);
        cv::Mat lastResizepose = mLastResizePose;;
        if(pRef->IsScaled()==false)
        {
            ResizePose(lastResizepose, *pRef, detectedscale);
            pRef->SetScaleFlag(); 
        }
        std::set<MapPoint*> refmappoints = pRef->GetMapPoints(); 
        for(set<MapPoint*>::iterator sit=refmappoints.begin(), send=refmappoints.end(); sit!=send; sit++)
        {
            if((*sit)->isBad() || (*sit)->IsScaled() )
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            ResizePoint(lastResizepose, (*sit), detectedscale);
            (*sit)->SetScaleFlag();
        }
        // mLastFrame needed to be reset cuz ref is reseted
        mLastFrame.SetPose(Tlr*pRef->GetPose()); 
        
        mLastResizeFrameID = mCurrentFrame.mnId;
        mLastResizePose = mLastFrame.mTcw.clone();
        mScaleEst.mLastResizeFrameID = mLastResizeFrameID;
        
        if(!mVelocity.empty()){
          // cout << "Start to resize mVelocity with scale: " << detectedscale << endl;
          cv::Mat tmp_t;
          mVelocity.col(3).rowRange(0,3).copyTo(tmp_t);
          tmp_t = tmp_t*detectedscale;
          tmp_t.copyTo(mVelocity.col(3).rowRange(0,3));
        }
        scaleinTracking = false;
    }

  if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR ||!mbOnlyTracking)
    return;

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  vector<pair<float, int> > vDepthIdx;
  vDepthIdx.reserve(mLastFrame.N);
  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty()) return;

  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
  }
}

float Tracking::DistBetweenFrames(const Frame& lastFrame,
                                  const Frame& currentFrame) {
  // distance between 2 frames
  cv::Mat Tcw = currentFrame.mTcw.clone();
  cv::Mat Tcw0 = lastFrame.mTcw.clone();

  cv::Mat Rwc(3, 3, CV_32F);
  cv::Mat twc(3, 1, CV_32F);
  Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
  twc = -Rwc * Tcw.rowRange(0, 3).col(3);

  cv::Mat Rwc0(3, 3, CV_32F);
  cv::Mat twc0(3, 1, CV_32F);
  Rwc0 = Tcw0.rowRange(0, 3).colRange(0, 3).t();
  twc0 = -Rwc0 * Tcw0.rowRange(0, 3).col(3);

  // double dist_vo = sqrt( twc.at<float>(0,3)*twc.at<float>(0,3) +
  // twc.at<float>(1,3)*twc.at<float>(1,3) +
  // twc.at<float>(2,3)*twc.at<float>(2,3)); double dist_vo = sqrt(
  // Tcw.at<float>(0,3)*Tcw.at<float>(0,3) +
  // Tcw.at<float>(1,3)*Tcw.at<float>(1,3) +
  // Tcw.at<float>(2,3)*Tcw.at<float>(2,3)); double dx = Tcw.at<float>(0,3) -
  // mTcw0.at<float>(0,3); double dy = Tcw.at<float>(1,3) -
  // mTcw0.at<float>(1,3); double dz = Tcw.at<float>(2,3) -
  // mTcw0.at<float>(2,3);

  /*    cout<<Tcw <<endl;
      cout<<Rwc<<endl;
      cout<<twc<<endl;
      cout<<Tcw0 <<endl;
      cout<<Rwc0<<endl;
      cout<<twc0<<endl;
  */
  float dx = (twc.at<float>(0, 0) - twc0.at<float>(0, 0));
  float dy = (twc.at<float>(1, 0) - twc0.at<float>(1, 0));
  float dz = (twc.at<float>(2, 0) - twc0.at<float>(2, 0));

  //    cout<< dx << ","<<dy<<","<<dz<<endl;
  double dist_vo = sqrt(dx * dx + dy * dy + dz * dz);
  //    cout<< "dist_vo = "<<dist_vo<<endl;

  return dist_vo;
}

void Tracking::ResizePose(const cv::Mat Tcw0, Frame &currentFrame, float scale){
  
  cv::Mat Tcw = currentFrame.mTcw.clone();
  cv::Mat Tcw_new;

  cv::Mat Rwc(3, 3, CV_32F);
  cv::Mat twc(3, 1, CV_32F);
  Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
  twc = -Rwc * Tcw.rowRange(0, 3).col(3);

  cv::Mat Rwc0(3, 3, CV_32F);
  cv::Mat twc0(3, 1, CV_32F);
  Rwc0 = Tcw0.rowRange(0, 3).colRange(0, 3).t();
  twc0 = -Rwc0 * Tcw0.rowRange(0, 3).col(3);

  cv::Mat S = cv::Mat::eye(4, 4, CV_32F);
  S.at<float>(0, 0) = scale;
  S.at<float>(1, 1) = scale;
  S.at<float>(2, 2) = scale;

  cv::Mat Tcw_tmp = Tcw * Tcw0.inv();
  Tcw_tmp.col(3).rowRange(0, 3) = Tcw_tmp.col(3).rowRange(0, 3) * scale;
  Tcw_new = Tcw_tmp * Tcw0;

  currentFrame.SetPose(Tcw_new);
}

/**
 * @brief resize KF relative to the last frame
 *
 * @param lastFrame: the last frame
 * @param currentFrame: the current frame
 * @param scale: scale factor
 */
void Tracking::ResizePose(const cv::Mat Tcw0, KeyFrame &currentFrame, float scale){
  cv::Mat Tcw = currentFrame.GetPose();
  cv::Mat Tcw_new;

  cv::Mat Tcw_tmp = Tcw * Tcw0.inv();
  Tcw_tmp.col(3).rowRange(0, 3) = Tcw_tmp.col(3).rowRange(0, 3) * scale;
  Tcw_new = Tcw_tmp * Tcw0;

  currentFrame.SetPose(Tcw_new);
}

/**
 * @brief resize key points with scale factor relative to the last frame
 *
 * @param lastFrame: the last frame of the last resizing
 * @param pMP: map point
 * @param scale: scale factor`
 */
void Tracking::ResizePoint(const cv::Mat Tcw0, MapPoint* pMP, float scale){

  // scale matrix
  cv::Mat S = cv::Mat::eye(4, 4, CV_32F);
  S.at<float>(0, 0) = scale;
  S.at<float>(1, 1) = scale;
  S.at<float>(2, 2) = scale;

  // to homogeneous coordinate
  cv::Mat Pw =
      pMP->GetWorldPos();     // map point in world coordinate (3x1 vector)
  cv::Mat PcH(4, 1, CV_32F);  // map point in camera coordinate. Homogeneous
                              // coordinate (4x1 vector)
  PcH.at<float>(0) = Pw.at<float>(0);
  PcH.at<float>(1) = Pw.at<float>(1);
  PcH.at<float>(2) = Pw.at<float>(2);
  PcH.at<float>(3) = 1;

  // world to camera
  cv::Mat Pc = Tcw0 * PcH;

  // scale
  cv::Mat Pnew = S * Pc;

  // camera to world
  cv::Mat PwH(4, 1, CV_32F);  // map point in world coordinate. Homogeneous
                              // coordinate (4x1 vector)
  PwH = Tcw0.inv() * Pnew;

  Pw.at<float>(0) = PwH.at<float>(0) / PwH.at<float>(3);
  Pw.at<float>(1) = PwH.at<float>(1) / PwH.at<float>(3);
  Pw.at<float>(2) = PwH.at<float>(2) / PwH.at<float>(3);

  pMP->SetWorldPos(Pw);

  pMP->UpdateNormalAndDepth();
}

bool Tracking::TrackWithTFPose(){
  std::cout << __FUNCTION__ << std::endl;
  ORBmatcher matcher(0.8, true);
  std::cout << "Last Frame pose = "<< std::endl << mLastFrame.mTcw << std::endl;
  UpdateLastFrame();

  if(mDR_Tcw.empty() || !(mCurrentFrame.mOdomFlag && mLastFrame.mOdomFlag))
    return false;
  
  mCurrentFrame.SetPose(mDR_Tcw);
  std::cout << "Current  pose: " << mCurrentFrame.mTcw << std::endl;

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  int th;
  if (mSensor != System::STEREO)
    th = mSearchWindowSize;  // before : 15;
  else
    th = 7;
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

  if (nmatches < mnMatches)  // default: 20
  {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
  }

  if (nmatches < mnMatches)  
    return false;

  Optimizer::PoseOutliner(&mCurrentFrame);

    // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > mnMatches;  // default: 20;
  }

  return nmatchesMap >= mnMatches;  // default: 20; // DR!!! before: 10
}

bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();
  if(mCurrentFrame.mOdomFlag && mLastFrame.mOdomFlag){
    mVelocity = CalculateVelocity();
  }

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  // Project points seen in previous frame
  int th;
  if (mSensor != System::STEREO)
    th = mSearchWindowSize;  // before : 15;
  else
    th = 7;
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                            mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < mnMatches)  // default: 20
  {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                          mSensor == System::MONOCULAR);
  }

  if (nmatches < mnMatches)  // default: 20 // DR!!! before: 20
    return false;

  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&mCurrentFrame);
 
  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        //TODO: 添加判断设置pMP->mbTrackInViewR的值

        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
        // ifdef SCALE_ESTIMATION
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > mnMatches;  // default: 20;
  }

   //cout<<"motion nmatchesMap: "<<nmatchesMap<<endl;

  return nmatchesMap >= mnMatches;  // default: 20; // DR!!! before: 10
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.
  //cout << "[Tracking]: " << __FUNCTION__<<endl;
  std::cout << __FUNCTION__ << std::endl;
  UpdateLocalMap();

  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;
  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else if (mSensor == System::STEREO)
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      mnMatchesInliers < 50)
    return false;

  if (mnMatchesInliers < mnThresTrackLocalMap)  // before: 30
    return false;
  else
    return true;
}

bool Tracking::NeedNewKeyFrame() {
  if (mbOnlyTracking) return false;

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 2;
  if (nKFs <= 2) nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be
  // potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;
  if (mSensor != System::MONOCULAR) {
    for (int i = 0; i < mCurrentFrame.N; i++) {
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
          nTrackedClose++;
        else
          nNonTrackedClose++;
      }
    }
  }

  bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2) thRefRatio = 0.4f;

  if (mSensor == System::MONOCULAR) thRefRatio = 0.9f;

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames &&
                    bLocalMappingIdle);
  // Condition 1c: tracking is weak
  const bool c1c =
      mSensor != System::MONOCULAR &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 =
      ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
       mnMatchesInliers > 15);
   //添加DR模式下的添加关键帧; DR模式下并且都有odom数据.也是当地添加关键帧
  if (mbDR && mCurrentFrame.mOdomFlag && mLastFrame.mOdomFlag) {
     //距离上一个关键帧已经过去20帧
     bool d0 = (mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId) > 20;
     // odom的行走距离已经查过一定数量或角度
     float odomDist = sqrt(
       (mCurrentFrame.mDrX - mLastFrame.mDrX) * 
       (mCurrentFrame.mDrX - mLastFrame.mDrX) +
       (mCurrentFrame.mDrY - mLastFrame.mDrY) *
       (mCurrentFrame.mDrY - mLastFrame.mDrY));
     bool d1 = odomDist >= 0.8;
     if (d1 || d1) return true;
  }
  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (mSensor != System::MONOCULAR) {
        if (mpLocalMapper->KeyframesInQueue() < 3)
          return true;
        else
          return false;
      } else
        return false;
    }
  } else
    return false;
}

bool Tracking::NeedNewKeyFrameDR()
{
  bool d1 = (mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId) > 30;
  cout << __FUNCTION__ ;
  cout << "mCurrentFrame  mDrX = " << mCurrentFrame.mDrX << "  mpLastKeyFrame->mDrX = "<<  mpLastKeyFrame->mDrX << std::endl;
  cout << "mCurrentFrame  mDrY = " << mCurrentFrame.mDrY << "  mpLastKeyFrame->mDrY = "<<  mpLastKeyFrame->mDrY << std::endl;

  float odom_dist = sqrt((mCurrentFrame.mDrX - mpLastKeyFrame->mDrX) *
                             (mCurrentFrame.mDrX - mpLastKeyFrame->mDrX) +
                         (mCurrentFrame.mDrY - mpLastKeyFrame->mDrY) *
                             (mCurrentFrame.mDrY - mpLastKeyFrame->mDrY));
  std::cout << __FUNCTION__ << " mCurrentFrame.mnId = " << mCurrentFrame.mnId << ", mpLastKeyFrame->mnId = " << mpLastKeyFrame->mnFrameId <<std::endl;
  std::cout << __FUNCTION__ << " odom_dist = " << odom_dist << std::endl;
  bool d2 = odom_dist > 0.8;

  if (d1 || d2)
    return true;
  else
    return false;
}

void Tracking::CreateNewKeyFrame() {
  if (!mpLocalMapper->SetNotStop(true)) return;

  KeyFrame* pKF =
      new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,  mSensor,mTbc);

  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();
    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;

  mpMap->IncreaseKFCounter();  // DR!!!
}

void Tracking::CreateNewKeyFrameDR() {
  if (!mpLocalMapper->SetNotStop(true)) return;
  cout << " CreateNewKeyFrameDR " ;
  KeyFrame* pKF =
      new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB, mSensor, mTbc);
  std::cout << " new False KF is " << pKF->mnId << std::endl;
  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;
        bool bCreateNew = false;
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }
        if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);
  mpLocalMapper->SetNotStop(false);
  pKF->AddDRKeyFrame(mpLastKeyFrame);  // DR!!!
  cout << __FUNCTION__ << " AddDRKeyFrame for " << mpLastKeyFrame->mnId << std::endl;
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
  mpMap->IncreaseKFCounter();  // DR!!!
  pKF->mbFalseKF = true;

  pKF->SetScaleFlag();

  if(mpInitializer)  //tmp: 防止各个关键帧之间交错
  {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
  }

  mLastResizeFrameID = mCurrentFrame.mnId;
  mLastResizePose = mLastFrame.mTcw.clone();
  mScaleEst.mLastResizeFrameID = mLastResizeFrameID;
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(),
                                   vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint*>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;
  // 验证其他的localPoints能不能在地图中被看到
  // Project points in frame and check its visibility
  for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                   vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) continue;
    if (pMP->isBad()) continue;
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustumFisheye(
            pMP, -0.131))  // FoV is 195 degrees , vn < cos(97.5)
    {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    ORBmatcher matcher(0.8);
    int th = 1;
    if (mSensor == System::RGBD) th = 3;
    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) th = 5;
    matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame* pKF = *itKF;
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                                           itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint* pMP = *itMP;
      if (!pMP) continue;
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  map<KeyFrame*, int> keyframeCounter;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
        for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(),
                                                    itend = observations.end();
             it != itend; it++)
          keyframeCounter[it->first]++;
      } else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty()) return;

  int max = 0;
  KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also
  // check which keyframe shares most points
  for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame* pKF = it->first;

    if (pKF->isBad()) continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // Include also some not-already-included keyframes that are neighbors to
  // already-included keyframes
  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80) break;

    KeyFrame* pKF = *itKF;

    const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                                           itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for (set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                                        send = spChilds.end();
         sit != send; sit++) {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  // Compute Bag of Words Vector
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  vector<KeyFrame*> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

  if (vpCandidateKFs.empty()) return false;

  const int nKFs = vpCandidateKFs.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<PnPsolver*> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  vector<vector<MapPoint*> > vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  for (int i = 0; i < nKFs; i++) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      int nmatches =
          matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        PnPsolver* pSolver =
            new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        vpPnPsolvers[i] = pSolver;
        nCandidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  ORBmatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    /*cout<<"nCandidates: "<<nCandidates<<endl;
    cout<<"bMatch: "<<bMatch<<endl;
    cout<<"nKFs: "<<nKFs<<endl;*/
    for (int i = 0; i < nKFs; i++) {
      // cout<<"vbDiscarded[i]: "<<vbDiscarded[i]<<endl;
      if (vbDiscarded[i]) {
        nCandidates--;
        continue;
      }

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      PnPsolver* pSolver = vpPnPsolvers[i];
      cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
      // cout<<"inKF: "<<i<<endl;
      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If a Camera Pose is computed, optimize
      if (!Tcw.empty()) {
        Tcw.copyTo(mCurrentFrame.mTcw);

        set<MapPoint*> sFound;

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else
            mCurrentFrame.mvpMapPoints[j] = NULL;
        }

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

        if (nGood < 10) {
          nCandidates--;
          continue;
        }

        for (int io = 0; io < mCurrentFrame.N; io++)
          if (mCurrentFrame.mvbOutlier[io])
            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

        // If few inliers, search by projection in a coarse window and optimize
        // again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(
              mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            // If many inliers but still not enough, search by projection again
            // in a narrower window the camera has been already optimized with
            // many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < mCurrentFrame.N; ip++)
                if (mCurrentFrame.mvpMapPoints[ip])
                  sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
              nadditional = matcher2.SearchByProjection(
                  mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                for (int io = 0; io < mCurrentFrame.N; io++)
                  if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] = NULL;
              }
            }
          }
        }

        // cout<<"nGood: "<<nGood<<endl;
        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        } else {
          vbDiscarded[i] = true;
          nCandidates--;
        }
      } else {
        vbDiscarded[i] = true;
        nCandidates--;
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    mLastResizeFrameID = mCurrentFrame.mnId;
    mLastResizePose = mLastFrame.mTcw.clone();
    return true;
  }
}

void Tracking::Reset() {
  cout << "System Reseting" << endl;
#ifdef VISUAL
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) usleep(3000);
  }
#endif
  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  mpLocalMapper->RequestReset();
  cout << " done" << endl;
#ifdef LC
  // Reset Loop Closing
  cout << "Reseting Loop Closing...";
  mpLoopClosing->RequestReset();
  cout << " done" << endl;
#endif
  // Clear BoW Database
  cout << "Reseting Database...";
  mpKeyFrameDB->clear();
  cout << " done" << endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = NO_IMAGES_YET;
  mbDR = false;
  mnLastRelocFrameId = 0;
  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer*>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();
#ifdef VISUAL

  if (mpViewer) mpViewer->Release();
#endif
}

void Tracking::ChangeCalibration(const string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool& flag) { mbOnlyTracking = flag; }

/**
 * @brief track with DR
 *
 */
cv::Mat Tracking::CalculateVelocity()
{
    cv::Mat Tb1b2 = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Rb1b2 = cv::Mat::eye(3,3,CV_32F);
    cv::Mat tb2inb1(3,1,CV_32F);
    float dth_rad = DR_del_th; 

    //cout << __FUNCTION__ << "dth_rad = " << dth_rad << endl;
    Rb1b2.at<float>(0,0) = cos(dth_rad);
    Rb1b2.at<float>(0,1) = -sin(dth_rad);
    Rb1b2.at<float>(1,0) = sin(dth_rad);
    Rb1b2.at<float>(1,1) = cos(dth_rad);
    float d = sqrt(DR_del_x*DR_del_x + DR_del_y*DR_del_y);
    float thc = atan2(DR_del_y, DR_del_x) - DR_th;
    float d_xr = d * cos(thc); //在baselink1的坐标系中baselink2的位移
    float d_yr = d * sin(thc);

    tb2inb1.at<float>(0) = d_xr;
    tb2inb1.at<float>(1) = d_yr;
    tb2inb1.at<float>(2) = 0;
    Rb1b2.copyTo(Tb1b2.rowRange(0,3).colRange(0,3));
    tb2inb1.copyTo(Tb1b2.rowRange(0,3).col(3));
    
    cv::Mat Tcb = mTcb.clone();
    cv::Mat Rcb = Tcb.colRange(0,3).rowRange(0,3).clone();
    cv::Mat tbinc = mbaselink_in_cam.clone();
  
    cv::Mat c2inb2 = mTbc.col(3).rowRange(0,3);
    cv::Mat c2inb1 = Rb1b2*c2inb2 + tb2inb1;
    cv::Mat c2inc1 = Rcb*c2inb1 + tbinc;
  
    //Rotation from c1 to c2, angle is -DR_del_th
    cv::Mat Rc1c2 = cv::Mat::eye(3,3,CV_32F);
    Rc1c2.at<float>(0,0) = cos(dth_rad);
    Rc1c2.at<float>(0,2) = sin(dth_rad);
    Rc1c2.at<float>(2,0) = -sin(dth_rad);
    Rc1c2.at<float>(2,2) = cos(dth_rad);
  
    cv::Mat Rc2c1 = Rc1c2.t();
    cv::Mat c1inc2 = -Rc2c1*c2inc1;
  
    cv::Mat Tc2c1 = cv::Mat::eye(4, 4, CV_32F);
    Rc2c1.copyTo(Tc2c1.colRange(0,3).rowRange(0,3));
    c1inc2.copyTo(Tc2c1.col(3).rowRange(0,3));
    
    return Tc2c1;
}

void Tracking::TrackWithDR()
{
    cv::Mat Tc1w = mTcw_1.clone();
    cv::Mat Tb2b1;
    cv::Mat Tb1b2 = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Rb1b2 = cv::Mat::eye(3,3,CV_32F);
    cv::Mat tb2inb1(3,1,CV_32F);
    float dth_rad = DR_del_th; 
    Rb1b2.at<float>(0,0) = cos(dth_rad);
    Rb1b2.at<float>(0,1) = -sin(dth_rad);
    Rb1b2.at<float>(1,0) = sin(dth_rad);
    Rb1b2.at<float>(1,1) = cos(dth_rad);
    // cout << "Rb1b2: " << Rb1b2 << endl;
    float d = sqrt(DR_del_x*DR_del_x + DR_del_y*DR_del_y);
    float thc = atan2(DR_del_y, DR_del_x) - DR_th;
    float d_xr = d * cos(thc); //在baselink1的坐标系中baselink2的位移
    float d_yr = d * sin(thc);

    tb2inb1.at<float>(0) = d_xr;
    tb2inb1.at<float>(1) = d_yr;
    tb2inb1.at<float>(2) = 0;
    Rb1b2.copyTo(Tb1b2.rowRange(0,3).colRange(0,3));
    tb2inb1.copyTo(Tb1b2.rowRange(0,3).col(3));
    // cout << "Tb1b2: " << Tb1b2 << endl;
    Tb2b1 = Tb1b2.inv();
    // cout << "Tb2b1: " << Tb2b1 << endl;

    cv::Mat new_Tbc = cv::Mat::eye(4,4,CV_32F);
    if(!mTbc.empty()){
         new_Tbc =mTbc.clone();
    }else{cout << "[Track] mTbc is empty " <<std::endl;}

    cv::Mat Tb1w = new_Tbc* Tc1w; 
    cv::Mat Tb2w = Tb2b1*Tb1w;
    cv::Mat Tc2w = new_Tbc.inv()*Tb2w;
    // cout << "Tc2w: " << Tc2w << endl; 
    mTcw_1 = Tc2w.clone();
    mDR_Tcw = mTcw_1.clone();

    mTcw_1.copyTo(mCurrentFrame.mTcw);
    #ifdef VISUAL
    mpMapDrawer->SetCurrentCameraPose(mDR_Tcw);
    #endif
}

void Tracking::SetFirstFramePose(float x, float y, float theta)
{
    cv::Mat Tc1w = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Tb2b1;
    cv::Mat Tb1b2 = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Rb1b2 = cv::Mat::eye(3,3,CV_32F);
    cv::Mat tb2inb1(3,1,CV_32F);
    float dth_rad = theta; 

    Rb1b2.at<float>(0,0) = cos(dth_rad);
    Rb1b2.at<float>(0,1) = -sin(dth_rad);
    Rb1b2.at<float>(1,0) = sin(dth_rad);
    Rb1b2.at<float>(1,1) = cos(dth_rad);
    // cout << "Rb1b2: " << Rb1b2 << endl;

    tb2inb1.at<float>(0) = x;
    tb2inb1.at<float>(1) = y;
    tb2inb1.at<float>(2) = 0;
    Rb1b2.copyTo(Tb1b2.rowRange(0,3).colRange(0,3));
    tb2inb1.copyTo(Tb1b2.rowRange(0,3).col(3));
    // cout << "Tb1b2: " << Tb1b2 << endl;
    Tb2b1 = Tb1b2.inv();
    // cout << "Tb2b1: " << Tb2b1 << endl;
    cv::Mat new_Tbc = cv::Mat::eye(4,4,CV_32F);
    if(!mTbc.empty()){
         new_Tbc =mTbc.clone();
    }else{cout << "[Track] mTbc is empty " <<std::endl;}
    
    // cout << "new Tbc: " << Tbc << endl;
    cv::Mat Tb1w = new_Tbc* Tc1w; 
    cv::Mat Tb2w = Tb2b1*Tb1w;
    cv::Mat Tc2w = new_Tbc.inv()*Tb2w;
    // cout << "Tc2w: " << Tc2w << endl; 
    mTcw_1 = Tc2w.clone();
    mDR_Tcw = mTcw_1.clone();

  #ifdef VISUAL
    mpMapDrawer->SetCurrentCameraPose(mDR_Tcw);
  #endif
    // cout << "mDR_Tcw: " << mDR_Tcw << endl;
}

/**
 * @brief resize map data (KFs and key points)
 *
 * @param sf: scale factor
 */
void Tracking::ResizeMapData(float sf) {
  cv::Mat lastResizepose = mLastResizePose;

  // scale KFs
  vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];

    if (pKF->IsScaled()) continue;

    float current_scale = pKF->mDistScale;
    ResizePose(lastResizepose, *pKF, sf);

    pKF->SetScaleFlag();
  }
  // cout << "FInished resize kf poses "<< endl;
  // scale points
  vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
  vector<MapPoint*> vpRefMPs = mpMap->GetReferenceMapPoints();

  set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

  if (vpMPs.empty()) return;

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]) || vpMPs[i]->IsScaled())
      continue;

        ResizePoint(lastResizepose, vpMPs[i], sf);

    vpMPs[i]->SetScaleFlag();
  }
  // cout << "FInished resize vpMPs "<< endl;

  for (set<MapPoint*>::iterator sit = spRefMPs.begin(), send = spRefMPs.end();
       sit != send; sit++) {
    if ((*sit)->isBad() || (*sit)->IsScaled()) continue;
    cv::Mat pos = (*sit)->GetWorldPos();

        ResizePoint(lastResizepose, (*sit), sf);

    (*sit)->SetScaleFlag();
  }
  
  // cout << "FInished resize mappoints "<< endl;
}

//针对突然的旋转或者晃动的修饰，重新使用DR计算的pose尝试跟踪
bool Tracking::ReTrackWithMotionModel(cv::Mat &DR_pose)
{
    // cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Using ReTrackWithMotionModel() "<< endl;
    ORBmatcher matcher(0.9,true);
    mCurrentFrame.SetPose(DR_pose);
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    int th = mSearchWindowSize*2; //直接使用最大的window进行重启
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    if(nmatches<mnMatches) // default: 20 // DR!!! before: 20
    {
        return false;
    }
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                //TODO: 添加判断设置pMP->mbTrackInViewR的值
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
           }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    std::cout << __FUNCTION__ << " nmatches is " << nmatches << endl;
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>mnMatches; // default: 20;
    }
      return nmatches >= mnMatches;  // default: 20; // DR!!! before: 10
}
}  // namespace ORB_SLAM2
