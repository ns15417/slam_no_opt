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

#include "LocalMapping.h"

#include <mutex>

#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#define LC
extern bool g_pause;

namespace ORB_SLAM2 {

LocalMapping::LocalMapping(Map *pMap, const float bMonocular)
    : mbMonocular(bMonocular),
      mbResetRequested(false),
      mbFinishRequested(false),
      mbFinished(true),
      mpMap(pMap),
      mbAbortBA(false),
      mbStopped(false),
      mbStopRequested(false),
      mbNotStop(false),
      mbAcceptKeyFrames(true) {}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
  mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

void LocalMapping::Run() {
  mbFinished = false;
  std::chrono::steady_clock::time_point oldt = std::chrono::steady_clock::now();
  while (1) {
    std::chrono::steady_clock::time_point newt =
        std::chrono::steady_clock::now();
    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double> >(newt - oldt)
            .count();
    if (ttrack > 1.0) {
      // cout << "Running LM " << endl;
      oldt = newt;
    }

    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      //cout << "[LocalMapping] New KF coming" << endl;
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // scale estimation
      // cout<<"Robot center = "<<mpCurrentKeyFrame->GetRobotCenter()<<endl;
      mpTracker->mScaleEst.updateDRandVO(mpCurrentKeyFrame->mDrX,
                                         mpCurrentKeyFrame->mDrY,
                                         mpCurrentKeyFrame->GetRobotCenter(),
                                         mpCurrentKeyFrame->mDistScale);

      // Check recent MapPoints
      MapPointCulling();

      // Triangulate new MapPoints
      CreateNewMapPoints();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      mbAbortBA = false;
      std::cout << "[ProcessNewKeyFrame] mpCurrentKeyFrame id = " << mpCurrentKeyFrame->mnId << std::endl;
      //如果mlNewKeyFrames是空，且no need to stop
      if (!CheckNewKeyFrames() &&!stopRequested() && !mpCurrentKeyFrame->isFalseKF())   {
        //cout << "[LM] Goto LOCALBA " << endl;
        // Local BA
        if (mpMap->KeyFramesInMap() > 3 &&
            mpTracker->scaleinTracking == false && !mpCurrentKeyFrame->mbDRKF &&
            !mpCurrentKeyFrame->mbInitKF)
          Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA,
                                           mpMap);

        // Check redundant local Keyframes
        KeyFrameCulling();
      }
      
      mpTracker->mScaleEst.mCurrentKFID = mpCurrentKeyFrame->mnFrameId;
      mpTracker->mScaleEst.mCurrentKFOdomFlag = mpCurrentKeyFrame->mKFOdomFlag;
      bool isDrifted = mpTracker->mScaleEst.checkScaleDrift();
      if (isDrifted) {
        cout << "[LM] drift detected" << endl;
        float scale = mpTracker->mScaleEst.getScale();
        mpTracker->detectedscale = scale;
        mpTracker->scaleinTracking = true;
        mpTracker->mScaleEst.resetParams();
        //mpTracker->IsScaled = false;  // to resize map
      }

#ifdef LC
      std::cout << "[ProcessNewKeyFrame] mpCurrentKeyFrame id = " << mpCurrentKeyFrame->mnId << std::endl;
      if(!mpCurrentKeyFrame) std::cout << "mpCurrentKeyFrame is empty " << std::endl;
      mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
#endif
    } else if (Stop()) {
      // Safe area to stop
      while (isStopped() && !CheckFinish()) {
        usleep(3000);
      }
      if (CheckFinish()) break;
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(true);

    if (CheckFinish()) break;

    usleep(3000);
  }

  SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexNewKFs);
  mlNewKeyFrames.push_back(pKF);
  mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexNewKFs);
  return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame() {
  {
    unique_lock<mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    std::cout << "[ProcessNewKeyFrame] mpCurrentKeyFrame id = " << mpCurrentKeyFrame->mnId << std::endl;
    mlNewKeyFrames.pop_front();
  }

  // Compute Bags of Words structures
  mpCurrentKeyFrame->ComputeBoW();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();

  for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
          pMP->AddObservation(mpCurrentKeyFrame, i);
          pMP->UpdateNormalAndDepth();
          pMP->ComputeDistinctiveDescriptors();
        } else  // this can only happen for new stereo points inserted by the
                // Tracking
        {
          mlpRecentAddedMapPoints.push_back(pMP);
        }
      }
    }
  }

  // Update links in the Covisibility Graph
  mpCurrentKeyFrame->UpdateConnections();

  // Insert Keyframe in Map
  mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling() {
  // Check Recent Added MapPoints
  list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

  int nThObs;
  if (mbMonocular)
    nThObs = 2;
  else
    nThObs = 3;
  const int cnThObs = nThObs;

  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pMP = *lit;
    if (pMP->isBad()) {
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundRatio() < 0.25f) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 &&
               pMP->Observations() <= cnThObs) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
      lit = mlpRecentAddedMapPoints.erase(lit);
    else
      lit++;
  }
}

void LocalMapping::CreateNewMapPoints() {
  // Retrieve neighbor keyframes in covisibility graph
  int nn = 10;
  if (mbMonocular) nn = 20;
  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  ORBmatcher matcher(0.6, false);

  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3, 4, CV_32F);
  Rcw1.copyTo(Tcw1.colRange(0, 3));
  tcw1.copyTo(Tcw1.col(3));
  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

  int nnew = 0;

  // Search matches with epipolar restriction and triangulate
  for (size_t i = 0; i < vpNeighKFs.size(); i++) {
    if (i > 0 && CheckNewKeyFrames()) return;

    KeyFrame *pKF2 = vpNeighKFs[i];
    GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    const float baseline = cv::norm(vBaseline);
    if(!pKF2->mbFalseKF){
      if (!mbMonocular) {
        if (baseline < pKF2->mb) continue;
      }
      else {
        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepth = baseline / medianDepthKF2;

        if (ratioBaselineDepth < 0.01) continue;
        }
    }

    // Compute Fundamental Matrix
    cv::Mat E12 = ComputeF12(mpCurrentKeyFrame, pKF2);  // E12

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t, size_t> > vMatchedIndices;
    matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, E12,
                                   vMatchedIndices, false);

    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;
      const int &idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
      const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
      bool bStereo1 = kp1_ur >= 0;
      const cv::Point3f &P3M1 = mpCurrentKeyFrame->mvP3M[idx1];
      cv::Point3f P3M1_r;
      if(bStereo1)
          P3M1_r = mpCurrentKeyFrame->mvP3MRight[mpCurrentKeyFrame->mvMatcheslr[idx1]];

      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
      const float kp2_ur = pKF2->mvuRight[idx2];
      bool bStereo2 = kp2_ur >= 0;
      const cv::Point3f &P3M2 = pKF2->mvP3M[idx2];
      cv::Point3f P3M2_r;
      if(bStereo2)
          P3M2_r = pKF2->mvP3MRight[pKF2->mvMatcheslr[idx2]];

      cv::Mat xn1 = (cv::Mat_<float>(3, 1) << P3M1.x, P3M1.y, P3M1.z);
      cv::Mat xn2 = (cv::Mat_<float>(3, 1) << P3M2.x, P3M2.y, P3M2.z);

      cv::Mat ray1 = Rwc1 * xn1;
      cv::Mat ray2 = Rwc2 * xn2;
      const float cosParallaxRays =
          ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      if (bStereo1)
        cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2,
                                           mpCurrentKeyFrame->mvDepth[idx1]));
      else if (bStereo2)
        cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      cv::Mat x3D;
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
        // Linear Triangulation Method
        cv::Mat A(4, 4, CV_32F);
        A.row(0) =
            xn1.at<float>(0) * Tcw1.row(2) - xn1.at<float>(2) * Tcw1.row(0);
        A.row(1) =
            xn1.at<float>(1) * Tcw1.row(2) - xn1.at<float>(2) * Tcw1.row(1);
        A.row(2) =
            xn2.at<float>(0) * Tcw2.row(2) - xn2.at<float>(2) * Tcw2.row(0);
        A.row(3) =
            xn2.at<float>(1) * Tcw2.row(2) - xn2.at<float>(2) * Tcw2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0) continue;

        // Euclidean coordinates
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

      } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
      } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        x3D = pKF2->UnprojectStereo(idx2);
      } else
        continue;  // No stereo and very low parallax

      cv::Mat x3Dt = x3D.t();

      // Check triangulation in front of cameras
      float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
      if (z1 <= 0) continue;

      float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
      if (z2 <= 0) continue;

      // Check reprojection error in first keyframe
      const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
      const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
      const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
      cv::Mat x3Dc1(1, 3, CV_32F);
      x3Dc1.at<float>(0) = x1;
      x3Dc1.at<float>(1) = y1;
      x3Dc1.at<float>(2) = z1;

      if (!bStereo1) {//MONO
        cv::Point2f imguv;
        int ret = pCamera1->world2Img(x3Dc1,imguv);

        if (ret == -1) {
          cout << "out of the range" << endl;
          continue;
        }
        //reprojection error
        float errX1 = imguv.x - kp1.pt.x;
        float errY1 = imguv.y - kp1.pt.y;

        if ((errX1 * errX1 + errY1 * errY1)> 5.991 * sigmaSquare1)
          continue;
      } 
      else 
      {  // STEREO
        GeometricCamera *pCamera1_r = mpCurrentKeyFrame->mpCamera2;
        float x1_r = mpCurrentKeyFrame->mRrl.row(0).dot(x3Dc1) +
                     mpCurrentKeyFrame->mtlinr.at<float>(0);
        float y1_r = mpCurrentKeyFrame->mRrl.row(1).dot(x3Dc1) +
                     mpCurrentKeyFrame->mtlinr.at<float>(1);
        float z1_r = mpCurrentKeyFrame->mRrl.row(2).dot(x3Dc1) +
                     mpCurrentKeyFrame->mtlinr.at<float>(2);
        
        cv::Mat x3Dc1_r(3,1,CV_32F);
        x3Dc1_r.at<float>(0) = x1_r; x3Dc1_r.at<float>(1) = y1_r;  x3Dc1_r.at<float>(2) = z1_r; 
        cv::Point2f imguv1, imguv2;
        int ret1 = pCamera1->world2Img(x3Dc1, imguv1);
        int ret2 = pCamera1_r->world2Img(x3Dc1_r, imguv2);
  
        // EUCM model unprojection range, R2 = Mx^2+My^2
        if (ret1 == -1 || ret2 == -1) {
          continue;
        }

        float errX1 = imguv1.x - kp1.pt.x;
        float errY1 = imguv1.y - kp1.pt.y;
        float errX1_r = imguv2.x - kp1_ur;

        if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
          continue;
      }

      // Check reprojection error in second keyframe
      const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
      const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
      const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);

      cv::Mat x3Dc2(1, 3, CV_32F);
      x3Dc2.at<float>(0) = x2;
      x3Dc2.at<float>(1) = y2;
      x3Dc2.at<float>(2) = z2;
      if (!bStereo2) {//MONO
        cv::Point2f imguv;
        int ret = pCamera2->world2Img(x3Dc2,imguv);
        if (ret == -1) {
          cout << "out of the range" << endl;
          continue;
        }

        float errX2 = imguv.x - kp2.pt.x;
        float errY2 = imguv.y - kp2.pt.y;

        if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
          continue;
      }
      else 
      {//STEREO
        GeometricCamera *pCamera2_r = pKF2->mpCamera2;
        float x2_r = pKF2->mRrl.row(0).dot(x3Dc2) + pKF2->mtlinr.at<float>(0);
        float y2_r = pKF2->mRrl.row(1).dot(x3Dc2) + pKF2->mtlinr.at<float>(1);
        float z2_r = pKF2->mRrl.row(2).dot(x3Dc2) + pKF2->mtlinr.at<float>(2);
        
        cv::Mat x3Dc2_r(3,1,CV_32F);
        x3Dc2_r.at<float>(0) = x2_r; x3Dc2_r.at<float>(1) = y2_r; x3Dc2_r.at<float>(2) = z2_r;

        cv::Point2f imguv1, imguv2;
        int right_ret1 = pCamera2->world2Img(x3Dc2,imguv1);
        int right_ret2 = pCamera2->world2Img(x3Dc2_r, imguv2);

        if (right_ret1 == -1 || right_ret2 == -1) {
          continue;
        }

        float errX2 = imguv1.x - kp2.pt.x;
        float errY2 = imguv1.y - kp2.pt.y;
        float errX2_r = imguv2.x - kp2_ur;
        if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
          continue;
      }

      // Check scale consistency
      cv::Mat normal1 = x3D - Ow1;
      float dist1 = cv::norm(normal1);

      cv::Mat normal2 = x3D - Ow2;
      float dist2 = cv::norm(normal2);

      if (dist1 == 0 || dist2 == 0) continue;

      const float ratioDist = dist2 / dist1;
      const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] /
                                pKF2->mvScaleFactors[kp2.octave];

      /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
          continue;*/
      if (ratioDist * ratioFactor < ratioOctave ||
          ratioDist > ratioOctave * ratioFactor)
        continue;

      // Triangulation is succesfull
      MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

      pMP->AddObservation(mpCurrentKeyFrame, idx1);
      pMP->AddObservation(pKF2, idx2);

      mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
      pKF2->AddMapPoint(pMP, idx2);

      pMP->ComputeDistinctiveDescriptors();

      pMP->UpdateNormalAndDepth();

      mpMap->AddMapPoint(pMP);
      mlpRecentAddedMapPoints.push_back(pMP);

      nnew++;
    }
  }
}

void LocalMapping::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  int nn = 10;
  if (mbMonocular) nn = 20;
  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
  vector<KeyFrame *> vpTargetKFs;
  for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(),
                                          vend = vpNeighKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;
    if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
      continue;
    vpTargetKFs.push_back(pKFi);
    pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

    // Extend to some second neighbors
    const vector<KeyFrame *> vpSecondNeighKFs =
        pKFi->GetBestCovisibilityKeyFrames(5);
    for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(),
                                            vend2 = vpSecondNeighKFs.end();
         vit2 != vend2; vit2++) {
      KeyFrame *pKFi2 = *vit2;
      if (pKFi2->isBad() ||
          pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
          pKFi2->mnId == mpCurrentKeyFrame->mnId)
        continue;
      vpTargetKFs.push_back(pKFi2);
    }
  }

  // Search matches by projection from current KF in target KFs
  ORBmatcher matcher;
  vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();
  for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(),
                                    vend = vpTargetKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;

    matcher.Fuse(pKFi, vpMapPointMatches);
  }

  // Search matches by projection from target KFs in current KF
  vector<MapPoint *> vpFuseCandidates;
  vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

  for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(),
                                    vendKF = vpTargetKFs.end();
       vitKF != vendKF; vitKF++) {
    KeyFrame *pKFi = *vitKF;

    vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

    for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(),
                                      vendMP = vpMapPointsKFi.end();
         vitMP != vendMP; vitMP++) {
      MapPoint *pMP = *vitMP;
      if (!pMP) continue;
      if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
        continue;
      pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
      vpFuseCandidates.push_back(pMP);
    }
  }

  matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

  // Update points
  vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
      }
    }
  }

  // Update connections in covisibility graph
  mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  cv::Mat R12 = R1w * R2w.t();
  cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  cv::Mat t12x = SkewSymmetricMatrix(t12);

  // const cv::Mat &K1 = pKF1->mK;
  // const cv::Mat &K2 = pKF2->mK;

  // return K1.t().inv()*t12x*R12*K2.inv();
  return t12x * R12;
}

void LocalMapping::RequestStop() {
  //cout << "shouting requested " << endl;
  unique_lock<mutex> lock(mMutexStop);
  mbStopRequested = true;
  unique_lock<mutex> lock2(mMutexNewKFs);
  mbAbortBA = true;
}

bool LocalMapping::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  if (mbStopRequested && !mbNotStop) {
    mbStopped = true;
    //cout << "Heard Local Mapping STOP" << endl;
    return true;
  }

  return false;
}

bool LocalMapping::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

bool LocalMapping::stopRequested() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopRequested;
}

void LocalMapping::Release() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);
  if (mbFinished) return;
  mbStopped = false;
  mbStopRequested = false;
  for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(),
                                  lend = mlNewKeyFrames.end();
       lit != lend; lit++)
  {
    if ((*lit)->mbInitKF)
      continue;
    delete *lit;
  }

  mlNewKeyFrames.clear();

  //cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames() {
  unique_lock<mutex> lock(mMutexAccept);
  return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
  unique_lock<mutex> lock(mMutexAccept);
  mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag) {
  unique_lock<mutex> lock(mMutexStop);

  if (flag && mbStopped) return false;

  mbNotStop = flag;

  return true;
}

void LocalMapping::InterruptBA() { mbAbortBA = true; }

void LocalMapping::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are
  // seen in at least other 3 keyframes (in the same or finer scale) We only
  // consider close stereo points
  vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(),
                                    vend = vpLocalKeyFrames.end();
       vit != vend; vit++) {
    KeyFrame *pKF = *vit;
    if (pKF->mbInitKF == true) continue;
    const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

    int nObs = 3;
    const int thObs = nObs;
    int nRedundantObservations = 0;
    int nMPs = 0;
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          if (!mbMonocular) {
            if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
              continue;
          }

          nMPs++;
          if (pMP->Observations() > thObs) {
            const int &scaleLevel = pKF->mvKeysUn[i].octave;
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            int nObs = 0;
            for (map<KeyFrame *, size_t>::const_iterator
                     mit = observations.begin(),
                     mend = observations.end();
                 mit != mend; mit++) {
              KeyFrame *pKFi = mit->first;
              if (pKFi == pKF) continue;
              const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

              if (scaleLeveli <= scaleLevel + 1) {
                nObs++;
                if (nObs >= thObs) break;
              }
            }
            if (nObs >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > 0.9 * nMPs) pKF->SetBadFlag();
  }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
  return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
          v.at<float>(2), 0, -v.at<float>(0), -v.at<float>(1), v.at<float>(0),
          0);
}

void LocalMapping::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested) break;
    }
    usleep(3000);
  }
}

void LocalMapping::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlNewKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
    mbResetRequested = false;
  }
}

void LocalMapping::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool LocalMapping::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LocalMapping::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
  unique_lock<mutex> lock2(mMutexStop);
  mbStopped = true;
}

bool LocalMapping::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

}  // namespace ORB_SLAM2
