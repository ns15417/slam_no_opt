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

#include "Optimizer.h"

#include <Eigen/StdVector>
#include <mutex>

#include "Converter.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include "OptimizableTypes.h"

extern bool g_pause;

namespace ORB_SLAM2 {

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations,
                                       bool bFixInitPos, bool* pbStopFlag,
                                       const unsigned long nLoopKF,
                                       const bool bRobust) {
  vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
  vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
  BundleAdjustment(vpKFs, vpMP, nIterations, bFixInitPos, pbStopFlag, nLoopKF,
                   bRobust);
}

void Optimizer::BundleAdjustmentOriginal(const vector<KeyFrame*>& vpKFs,
                                 const vector<MapPoint*>& vpMP, int nIterations,
                                 bool bFixInitPos, bool* pbStopFlag,
                                 const unsigned long nLoopKF,
                                 const bool bRobust) {
  cv::Mat Rrl = vpKFs[0]->mRrl;
  cv::Mat tlinr = vpKFs[0]->mtlinr;
  Eigen::Matrix3d R21;
  Eigen::Vector3d t2;
  if (!Rrl.empty()) {
    R21 = Converter::toMatrix3d(Rrl);
    t2 = Converter::toVector3d(tlinr);
  }

  vector<bool> vbNotIncludedMP;
  vbNotIncludedMP.resize(vpMP.size());

  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

  linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);

  long unsigned int maxKFid = 0;

  // Set KeyFrame vertices
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
    vSE3->setId(pKF->mnId);
    vSE3->setFixed(pKF->mbInitKF==true);

    optimizer.addVertex(vSE3);
    if (pKF->mnId > maxKFid) maxKFid = pKF->mnId;
  }

  const float thHuber2D = sqrt(5.99);
  const float thHuber4D = sqrt(9.49);

  // Set MapPoint vertices
  for (size_t i = 0; i < vpMP.size(); i++) {
    MapPoint* pMP = vpMP[i];
    if (pMP->isBad()) continue;
    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
    const int id = pMP->mnId + maxKFid + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    const map<KeyFrame*, size_t> observations = pMP->GetObservations();

    int nEdges = 0;
    // SET EDGES
    for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin();
         mit != observations.end(); mit++) {
      KeyFrame* pKF = mit->first;
      if (pKF->isBad() || pKF->mnId > maxKFid) continue;

      nEdges++;

      const cv::KeyPoint& kpUn = pKF->mvKeysUn[mit->second];

      if (pKF->mvuRight[mit->second] < 0) {
        Eigen::Matrix<double, 2, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(pKF->mnId)));
        e->setMeasurement(obs);
        const float& invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        if (bRobust) {
          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuber2D);
        }

        e->fx = pKF->fx;
        e->fy = pKF->fy;
        e->cx = pKF->cx;
        e->cy = pKF->cy;
        e->alpha = (pKF->mDistCoef).at<float>(0);
        e->beta = (pKF->mDistCoef).at<float>(1);

        optimizer.addEdge(e);
      } else {
        Eigen::Matrix<double, 4, 1> obs;
        const float kp_ur = pKF->mvuRight[mit->second];
        const float kp_vr = pKF->mvvRight[mit->second];
        obs << kpUn.pt.x, kpUn.pt.y, kp_ur,kp_vr;

        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(pKF->mnId)));
        e->setMeasurement(obs);
        const float& invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
        Eigen::Matrix4d Info = Eigen::Matrix4d::Identity() * invSigma2;
        e->setInformation(Info);

        if (bRobust) {
          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuber4D);
        }

        e->fx = pKF->fx;
        e->fy = pKF->fy;
        e->cx = pKF->cx;
        e->cy = pKF->cy;
        e->alpha = (pKF->mDistCoef).at<float>(0);
        e->beta = (pKF->mDistCoef).at<float>(1);
        e->r_fx = pKF->mpCamera2->mvParameters[0];
        e->r_fy = pKF->mpCamera2->mvParameters[1];
        e->r_cx = pKF->mpCamera2->mvParameters[2];
        e->r_cy = pKF->mpCamera2->mvParameters[3];
        e->r_alpha = pKF->mpCamera2->mvParameters[4];
        e->r_beta = pKF->mpCamera2->mvParameters[5];
        e->R21 = R21;
        e->t2 = t2;

        optimizer.addEdge(e);
      }
    }

    if (nEdges == 0) {
      optimizer.removeVertex(vPoint);
      vbNotIncludedMP[i] = true;
    } else {
      vbNotIncludedMP[i] = false;
    }
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(nIterations);

  // Recover optimized data

  // find 1st Key frame
  cv::Mat comp;
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];

    if (pKF->mnId == 0) {
      g2o::VertexSE3Expmap* vSE3 =
          static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
      g2o::SE3Quat SE3quat = vSE3->estimate();

      cv::Mat p0 = Converter::toCvMat(SE3quat);
      comp = p0.inv();
      // cv::Mat res = comp * p0;
      // cout<< res <<endl;
    }
  }

  // for fix initial KF
  bool mbFixInit = bFixInitPos;

  // Keyframes
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;
    g2o::VertexSE3Expmap* vSE3 =
        static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    if (nLoopKF == 0) {
      if (!mbFixInit)
        pKF->SetPose(Converter::toCvMat(SE3quat));
      else
        pKF->SetPose(Converter::toCvMat(SE3quat) * comp);

    } else {
      pKF->mTcwGBA.create(4, 4, CV_32F);

      if (!mbFixInit)
        Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
      else {
        cv::Mat tmp = Converter::toCvMat(SE3quat) * comp;

        tmp.copyTo(pKF->mTcwGBA);
      }

      pKF->mnBAGlobalForKF = nLoopKF;
    }
  }

  // Points
  for (size_t i = 0; i < vpMP.size(); i++) {
    if (vbNotIncludedMP[i]) continue;

    MapPoint* pMP = vpMP[i];

    if (pMP->isBad()) continue;
    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
        optimizer.vertex(pMP->mnId + maxKFid + 1));

    if (nLoopKF == 0) {
      if (!mbFixInit)
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
      else {
        // cv::Mat pos = Converter::toCvMat(vPoint->estimate());
        // cv::Mat delta = comp.rowRange(0,3).colRange(3,4);
        // cout<<pos<<endl;
        // cout<<delta<<endl;

        // pMP->SetWorldPos( Converter::toCvMat(vPoint->estimate()) -
        // comp.rowRange(0,3).colRange(3,4));

        cv::Mat pos = Converter::toCvMat(vPoint->estimate());  // 3x1 vector
        cv::Mat posH(4, 1, CV_32F);
        posH.at<float>(0) = pos.at<float>(0);
        posH.at<float>(1) = pos.at<float>(1);
        posH.at<float>(2) = pos.at<float>(2);
        posH.at<float>(3) = 1;
        cv::Mat trans = comp.inv();
        cv::Mat pos_new = trans.rowRange(0, 3).colRange(0, 4) * posH;

        pMP->SetWorldPos(pos_new);
      }
      pMP->UpdateNormalAndDepth();
    } else {
      pMP->mPosGBA.create(3, 1, CV_32F);
      if (!mbFixInit)
        Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
      else {
        // cv::Mat pos = Converter::toCvMat(vPoint->estimate());
        // cv::Mat delta = comp.rowRange(0,3).colRange(3,4);
        // cout<<pos<<endl;
        // cout<<delta<<endl;

        // cv::Mat tmp = Converter::toCvMat(vPoint->estimate()) -
        // comp.rowRange(0,3).colRange(3,4); tmp.copyTo(pMP->mPosGBA);

        cv::Mat pos = Converter::toCvMat(vPoint->estimate());  // 3x1 vector
        cv::Mat posH(4, 1, CV_32F);
        posH.at<float>(0) = pos.at<float>(0);
        posH.at<float>(1) = pos.at<float>(1);
        posH.at<float>(2) = pos.at<float>(2);
        posH.at<float>(3) = 1;
        cv::Mat trans = comp.inv();
        cv::Mat pos_new = trans.rowRange(0, 3).colRange(0, 4) * posH;
        pos_new.copyTo(pMP->mPosGBA);
      }
      pMP->mnBAGlobalForKF = nLoopKF;
    }
  }
}
//TODO: 传入的参数需要和BundleAdjustmentOriginal()对应
void Optimizer::BundleAdjustment(const vector<KeyFrame*>& vpKFs,
                                 const vector<MapPoint*>& vpMP, int nIterations,
                                 bool bFixInitPos, bool* pbStopFlag,
                                 const unsigned long nLoopKF,
                                 const bool bRobust) {
  std::cout << "[OP] " << __FUNCTION__ << std::endl;
  vector<bool> vbNotIncludedMP;
  vbNotIncludedMP.resize(vpMP.size());

  // Map* pMap = vpKFs[0]->GetMap();

  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

  linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(false);

  if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);

  long unsigned int maxKFid = 0;

  const int nExpectedSize = (vpKFs.size()) * vpMP.size();
  //左目
  vector<ORB_SLAM2::EdgeSE3ProjectXYZ*> vpEdgesMono;
  vpEdgesMono.reserve(nExpectedSize);
  //右目
  vector<ORB_SLAM2::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
  vpEdgesBody.reserve(nExpectedSize);

  vector<KeyFrame*> vpEdgeKFMono;
  vpEdgeKFMono.reserve(nExpectedSize);

  vector<KeyFrame*> vpEdgeKFBody;
  vpEdgeKFBody.reserve(nExpectedSize);

  vector<MapPoint*> vpMapPointEdgeMono;
  vpMapPointEdgeMono.reserve(nExpectedSize);

  vector<MapPoint*> vpMapPointEdgeBody;
  vpMapPointEdgeBody.reserve(nExpectedSize);

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
    vSE3->setId(pKF->mnId);
    vSE3->setFixed(pKF->mbInitKF==true/*pKF->mnId == pMap->GetInitKFid()*/);
    optimizer.addVertex(vSE3);
    if (pKF->mnId > maxKFid) maxKFid = pKF->mnId;
  }

  const float thHuber2D = sqrt(5.99);
  const float thHuber3D = sqrt(7.815);

  for (size_t i = 0; i < vpMP.size(); i++) {
    MapPoint* pMP = vpMP[i];
    if (pMP->isBad()) continue;
    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
    const int id = pMP->mnId + maxKFid + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);
    //所有观测到这个地图点的关键帧
    const map<KeyFrame*, size_t> observations = pMP->GetObservations();

    int nEdges = 0;
    // SET EDGES
    for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin();
         mit != observations.end(); mit++) {
      KeyFrame* pKF = mit->first;
      if (pKF->isBad() || pKF->mnId > maxKFid) continue;
      if (optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
        continue;
      nEdges++;

      const int leftIndex = mit->second;  //该地图点在左目特征点序列中的ID

      if (leftIndex != -1 &&
          pKF->mvuRight[leftIndex] < 0)  //单目点, 该点在右目中对应的id
      {
        const cv::KeyPoint& kpUn = pKF->mvKeysUn[leftIndex];

        Eigen::Matrix<double, 2, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        ORB_SLAM2::EdgeSE3ProjectXYZ* e = new ORB_SLAM2::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(pKF->mnId)));
        e->setMeasurement(obs);
        const float& invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        if (bRobust) {
          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuber2D);
        }

        e->pCamera = pKF->mpCamera;

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vpEdgeKFMono.push_back(pKF);
        vpMapPointEdgeMono.push_back(pMP);
      }

      //适用于鱼眼相机, 左目相机和右目相机有单独的相机模型
      if (pKF->mpCamera2) {  //单独优化右目的重投影误差
        int leftIndex = mit->second;
        int rightIndex = pKF->mvMatcheslr[leftIndex];  //获取右目的index

        if (rightIndex != -1) {
          Eigen::Matrix<double, 2, 1> obs;
          cv::KeyPoint kp_left = pKF->mvKeys[leftIndex];
          float rightkp_u = pKF->mvuRight[leftIndex];
          float rightkp_v = pKF->mvvRight[leftIndex];
          obs << rightkp_u, rightkp_v;

          ORB_SLAM2::EdgeSE3ProjectXYZToBody* e =
              new ORB_SLAM2::EdgeSE3ProjectXYZToBody();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(pKF->mnId)));
          e->setMeasurement(obs);
          const float& invSigma2 = pKF->mvInvLevelSigma2[kp_left.octave];
          e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuber2D);

          e->mTrl = Converter::toSE3Quat(pKF->mTrl);

          e->pCamera = pKF->mpCamera2;

          optimizer.addEdge(e);
          vpEdgesBody.push_back(e);
          vpEdgeKFBody.push_back(pKF);
          vpMapPointEdgeBody.push_back(pMP);
        }
      }
    }

    if (nEdges == 0) {
      optimizer.removeVertex(vPoint);
      vbNotIncludedMP[i] = true;
    } else {
      vbNotIncludedMP[i] = false;
    }
  }

  // Optimize!
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(nIterations);

  // Recover optimized data
  // TODO: 这里注意3中的关键帧ID与原先2的关键帧id会有所不同,
  // 3会侧重于查找对应的地图中的关键帧 Keyframes
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;
    g2o::VertexSE3Expmap* vSE3 =
        static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

    g2o::SE3Quat SE3quat = vSE3->estimate();
    if (nLoopKF == 0 /*pMap->GetOriginKF()->mnId*/) {
      pKF->SetPose(Converter::toCvMat(SE3quat));
    } else {
      pKF->mTcwGBA.create(4, 4, CV_32F);
      Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
      pKF->mnBAGlobalForKF = nLoopKF;

      cv::Mat mTwc = pKF->GetPoseInverse();
      cv::Mat mTcGBA_c = pKF->mTcwGBA * mTwc;
      cv::Vec3d vector_dist = mTcGBA_c.rowRange(0, 3).col(3);
      double dist = cv::norm(vector_dist);
      if (dist > 1) {
        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
          ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
          MapPoint* pMP = vpMapPointEdgeMono[i];
          KeyFrame* pKFedge = vpEdgeKFMono[i];

          if (pKF != pKFedge) {
            continue;
          }

          if (pMP->isBad()) continue;

          if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            numMonoBadPoints++;
          } else {
            numMonoOptPoints++;
            vpMonoMPsOpt.push_back(pMP);
          }
        }
      }
    }
  }

  // Points
  for (size_t i = 0; i < vpMP.size(); i++) {
    if (vbNotIncludedMP[i]) continue;

    MapPoint* pMP = vpMP[i];

    if (pMP->isBad()) continue;
    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
        optimizer.vertex(pMP->mnId + maxKFid + 1));

    if (nLoopKF == 0 /*pMap->GetOriginKF()->mnId*/) {
      pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
      pMP->UpdateNormalAndDepth();
    } else {
      pMP->mPosGBA.create(3, 1, CV_32F);
      Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
      pMP->mnBAGlobalForKF = nLoopKF;
    }
  }
}

/**
 * @brief 当前帧的位姿不变,只是统计outliner的数量
 * 
 */
int Optimizer::PoseOutliner(Frame* pFrame){
  std::cout << __FUNCTION__ << std::endl;
  cv::Mat Rrl = pFrame->mRrl;
  cv::Mat tlinr = pFrame->mtlinr;

  int nInitialCorrespondences = 0;

  // Set MapPoint vertices
  const int N = pFrame->N;
  const float deltaMono = sqrt(5.991);
  const float chi2Mono =  5.991;
  cv::Mat current_pose = pFrame->mTcw;
  int nBad = 0;
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint* pMP = pFrame->mvpMapPoints[i];
      cv::KeyPoint kpUn;
      if (pMP) {
          nInitialCorrespondences++;
          cv::Mat P3D = pMP->GetWorldPos();
          //std::cout << "Original point " << P3D << std::endl;
          cv::Mat p3Dc = pFrame->mRcw*P3D + pFrame->mtcw;
        // Monocular observation
        if (pFrame->mvuRight[i] < 0) {
          kpUn = pFrame->mvKeysUn[i];
          cv::Point2f kpun_left(kpUn.pt.x, kpUn.pt.y);
          //利用当前的位姿计算地图点的重投影误差
          pFrame->mpCamera->SetMeasurement(kpun_left);
          cv::Vec2d error = pFrame->mpCamera->ComputeError(p3Dc);
          const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
          const float left_chi = pFrame->mpCamera->chi2(invSigma2);

          if (left_chi > chi2Mono) {
            pFrame->mvbOutlier[i] = true;
            nBad++;
          } else {
            pFrame->mvbOutlier[i] = false;
          }
        } else  // Stereo observation
        {
          pFrame->mvbOutlier[i] = false;
          cv::Mat p3Dc_r = Rrl*p3Dc+tlinr;
          //暂定测量值是点在右目中的像素坐标
          float kpright_u = pFrame->mvuRight[i];
          float kpright_v = pFrame->mvvRight[i];
          cv::Point2f kpRight(kpright_u,kpright_v);
          pFrame->mpCamera2->SetMeasurement(kpRight);
          cv::Vec2d error_r = pFrame->mpCamera2->ComputeError(p3Dc_r);
          int right_id = pFrame->mvMatcheslr[i];
          const float invSigma2_r = pFrame->mvInvLevelSigma2[pFrame->mvKeysRight[right_id].octave];

          const float chi2 = pFrame->mpCamera2->chi2(invSigma2_r);
          if(chi2 > chi2Mono){//右目重投影误差过大
            pFrame->mvbOutlier[i]=true;
            nBad++;
          }else
          {
            pFrame->mvbOutlier[i]=false;
          }
        }
      }// end of current point
    }// end of the for loop
  }
  std::cout << "nInitialCorrespondences = " << nInitialCorrespondences
            << ",nBad = " << nBad << std::endl;
  return nInitialCorrespondences - nBad;
}

int Optimizer::PoseOptimizationOriginal(Frame* pFrame) {
  cv::Mat Rrl = pFrame->mRrl;
  cv::Mat tlinr = pFrame->mtlinr;
  // 添加左右目的相对位姿R21， t2，
  // 可以将坐标由左目相机坐标系下转换到右目的坐标系下
  Eigen::Matrix3d R21;
  Eigen::Vector3d t2;
  if (!Rrl.empty()) {
    R21 = Converter::toMatrix3d(Rrl);
    t2 = Converter::toVector3d(tlinr);
  }
  
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

  linearSolver =
      new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = pFrame->N;

  vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
  vector<size_t> vnIndexEdgeStereo;
  vpEdgesStereo.reserve(N);
  vnIndexEdgeStereo.reserve(N);

  const float deltaMono = sqrt(5.991);
  const float deltaStereo = sqrt(9.49);

  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint* pMP = pFrame->mvpMapPoints[i];
      if (pMP) {
        // Monocular observation
        if (pFrame->mvuRight[i] < 0) {
          nInitialCorrespondences++;
          pFrame->mvbOutlier[i] = false;

          Eigen::Matrix<double, 2, 1> obs;
          const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
          obs << kpUn.pt.x, kpUn.pt.y;

          g2o::EdgeSE3ProjectXYZOnlyPose* e =
              new g2o::EdgeSE3ProjectXYZOnlyPose();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(0)));
          e->setMeasurement(obs);
          const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
          e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(deltaMono);

          e->fx = pFrame->fx;
          e->fy = pFrame->fy;
          e->cx = pFrame->cx;
          e->cy = pFrame->cy;
          e->alpha = (pFrame->mDistCoef).at<float>(0);
          e->beta = (pFrame->mDistCoef).at<float>(1);
          cv::Mat Xw = pMP->GetWorldPos();
          e->Xw[0] = Xw.at<float>(0);
          e->Xw[1] = Xw.at<float>(1);
          e->Xw[2] = Xw.at<float>(2);

          optimizer.addEdge(e);

          vpEdgesMono.push_back(e);
          vnIndexEdgeMono.push_back(i);
        } else  // Stereo observation
        {
          nInitialCorrespondences++;
          pFrame->mvbOutlier[i] = false;

          // SET EDGE
          Eigen::Matrix<double, 4, 1> obs;
          const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
          const float& kp_ur = pFrame->mvuRight[i];
          //右目中的横纵坐标也作为优化项
          const float& kp_vr = pFrame->mvvRight[i];
          obs << kpUn.pt.x, kpUn.pt.y, kp_ur, kp_vr;

          g2o::EdgeStereoSE3ProjectXYZOnlyPose* e =
              new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(0)));
          e->setMeasurement(obs);
          const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
          Eigen::Matrix4d Info = Eigen::Matrix4d::Identity() * invSigma2;
          e->setInformation(Info);

          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(deltaStereo);

          e->fx = pFrame->fx;
          e->fy = pFrame->fy;
          e->cx = pFrame->cx;
          e->cy = pFrame->cy;
          e->alpha = (pFrame->mDistCoef).at<float>(0);
          e->beta = (pFrame->mDistCoef).at<float>(1);

          e->r_fx = pFrame->mpCamera2->mvParameters[0];
          e->r_fy = pFrame->mpCamera2->mvParameters[1];
          e->r_cx = pFrame->mpCamera2->mvParameters[2];
          e->r_cy = pFrame->mpCamera2->mvParameters[3];
          e->r_alpha = pFrame->mpCamera2->mvParameters[4];
          e->r_beta =  pFrame->mpCamera2->mvParameters[5];

          e->R21 = R21;
          e->t2 = t2;
          cv::Mat Xw = pMP->GetWorldPos();
          e->Xw[0] = Xw.at<float>(0);
          e->Xw[1] = Xw.at<float>(1);
          e->Xw[2] = Xw.at<float>(2);

          optimizer.addEdge(e);

          vpEdgesStereo.push_back(e);
          vnIndexEdgeStereo.push_back(i);
        }
      }
    }
  }

  if (nInitialCorrespondences < 3) return 0;

  // We perform 4 optimizations, after each optimization we classify observation
  // as inlier/outlier At the next optimization, outliers are not included, but
  // at the end they can be classified as inliers again.
  const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
  const float chi2Stereo[4] = {9.49, 9.49, 9.49, 9.49};
  const int its[4] = {10, 10, 10, 10};

  int nBad = 0;
  for (size_t it = 0; it < 4; it++) {
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    optimizer.initializeOptimization(0);//0表示edge level, 只有non-outliner,上一轮优化的的setLevel(0)才会被放入到优化中
    optimizer.optimize(its[it]);

    nBad = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
      g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

      const size_t idx = vnIndexEdgeMono[i];

      if (pFrame->mvbOutlier[idx]) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2 > chi2Mono[it]) {
        pFrame->mvbOutlier[idx] = true;
        e->setLevel(1);
        nBad++;
      } else {
        pFrame->mvbOutlier[idx] = false;
        e->setLevel(0);
      }

      if (it == 2) e->setRobustKernel(0);
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

      const size_t idx = vnIndexEdgeStereo[i];

      if (pFrame->mvbOutlier[idx]) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2 > chi2Stereo[it]) {
        pFrame->mvbOutlier[idx] = true;
        e->setLevel(1);
        nBad++;
      } else {
        e->setLevel(0);
        pFrame->mvbOutlier[idx] = false;
      }

      if (it == 2) e->setRobustKernel(0);
    }

    if (optimizer.edges().size() < 10) break;
  }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap* vSE3_recov =
      static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  pFrame->SetPose(pose);

  return nInitialCorrespondences - nBad;
}

int Optimizer::PoseOptimization(Frame* pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;
    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<ORB_SLAM2::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM2::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
      unique_lock<mutex> lock(MapPoint::mGlobalMutex);

      for (int i = 0; i < N; i++) {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if (pMP) {
          nInitialCorrespondences++;

          cv::KeyPoint kpUn;

          if (pFrame->mvuRight[i] < 0) {  // Left camera observation
            kpUn = pFrame->mvKeys[i];

            pFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double, 2, 1> obs;
            obs << kpUn.pt.x, kpUn.pt.y;

            ORB_SLAM2::EdgeSE3ProjectXYZOnlyPose* e =
                new ORB_SLAM2::EdgeSE3ProjectXYZOnlyPose();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->pCamera = pFrame->mpCamera;
            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vnIndexEdgeMono.push_back(i);
          } else {  // Right camera observation 双目的情况
            cv::KeyPoint kpUn_left = pFrame->mvKeys[i];
            double kp_u = pFrame->mvuRight[i];
            double kp_v = pFrame->mvvRight[i];

            Eigen::Matrix<double, 2, 1> obs;
            obs << kp_u, kp_v;

            pFrame->mvbOutlier[i] = false;

            ORB_SLAM2::EdgeSE3ProjectXYZOnlyPoseToBody* e =
                new ORB_SLAM2::EdgeSE3ProjectXYZOnlyPoseToBody();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->pCamera = pFrame->mpCamera2;
            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            e->mTrl = Converter::toSE3Quat(pFrame->mTrl);  //左目到右目的变换

            optimizer.addEdge(e);

            vpEdgesMono_FHR.push_back(e);
            vnIndexEdgeRight.push_back(i);
          }
        }
      }
    }

    //cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            ORB_SLAM2::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
        {
            ORB_SLAM2::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustmentOriginal(KeyFrame* pKF, bool* pbStopFlag,
                                      Map* pMap) {
                                        cout << __FUNCTION__ << endl;
  cv::Mat Rrl = pKF->mRrl;
  cv::Mat tlinr = pKF->mtlinr;
  Eigen::Matrix3d R21;
  Eigen::Vector3d t2;
  if (!Rrl.empty()) {
    R21 = Converter::toMatrix3d(Rrl);
    t2 = Converter::toVector3d(tlinr);
  }

  // Local KeyFrames: First Breath Search from Current Keyframe
  list<KeyFrame*> lLocalKeyFrames;

  lLocalKeyFrames.push_back(pKF);
  pKF->mnBALocalForKF = pKF->mnId;

  const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
  for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
    KeyFrame* pKFi = vNeighKFs[i];
    pKFi->mnBALocalForKF = pKF->mnId;
    if (!pKFi->isBad()) lLocalKeyFrames.push_back(pKFi);
  }

  // Local MapPoints seen in Local KeyFrames
  list<MapPoint*> lLocalMapPoints;
  for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                 lend = lLocalKeyFrames.end();
       lit != lend; lit++) {
    vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
    for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end();
         vit != vend; vit++) {
      MapPoint* pMP = *vit;
      if (pMP)
        if (!pMP->isBad())
          if (pMP->mnBALocalForKF != pKF->mnId) {
            lLocalMapPoints.push_back(pMP);
            pMP->mnBALocalForKF = pKF->mnId;
          }
    }
  }

  // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local
  // Keyframes
  list<KeyFrame*> lFixedCameras;
  for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                 lend = lLocalMapPoints.end();
       lit != lend; lit++) {
    map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
    for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                          mend = observations.end();
         mit != mend; mit++) {
      KeyFrame* pKFi = mit->first;

      if (pKFi->mnBALocalForKF != pKF->mnId &&
          pKFi->mnBAFixedForKF != pKF->mnId) {
        pKFi->mnBAFixedForKF = pKF->mnId;
        if (!pKFi->isBad())
          lFixedCameras.push_back(pKFi);
        else if (pKFi->isDRKF())  //||pKFi->IsScaled())  // DR!!! to fix scaled
                                  //KF after ResizeMap
          lFixedCameras.push_back(pKFi);
      }
    }
  }

  // Setup optimizer
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

  linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);

  unsigned long maxKFid = 0;

  // Set Local KeyFrame vertices
  for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                 lend = lLocalKeyFrames.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
    vSE3->setId(pKFi->mnId);
    vSE3->setFixed(pKFi->mbInitKF==true);

    //vSE3->setFixed(pKFi->mnId < 100);  // test!!!

    /*if (pKFi->isDRKF())  // temp for DR!!!
      vSE3->setFixed(true);*/

    // if (pKFi->IsScaled())  // temp for DR!!!
    //   vSE3->setFixed(true);

    if (pKFi->isDRKF())  // temp for DR!!!
    {
      vSE3->setFixed(true);
      cout<<__FUNCTION__ << " setFixed for KF: "<< pKFi->mnId << endl;
    }  

    optimizer.addVertex(vSE3);
    if (pKFi->mnId > maxKFid) maxKFid = pKFi->mnId;
  }

  // Set Fixed KeyFrame vertices
  for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(),
                                 lend = lFixedCameras.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
    vSE3->setId(pKFi->mnId);
    vSE3->setFixed(true);
    optimizer.addVertex(vSE3);
    if (pKFi->mnId > maxKFid) maxKFid = pKFi->mnId;
  }

  // Set MapPoint vertices
  const int nExpectedSize =
      (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

  vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
  vpEdgesMono.reserve(nExpectedSize);

  vector<KeyFrame*> vpEdgeKFMono;
  vpEdgeKFMono.reserve(nExpectedSize);

  vector<MapPoint*> vpMapPointEdgeMono;
  vpMapPointEdgeMono.reserve(nExpectedSize);

  vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
  vpEdgesStereo.reserve(nExpectedSize);

  vector<KeyFrame*> vpEdgeKFStereo;
  vpEdgeKFStereo.reserve(nExpectedSize);

  vector<MapPoint*> vpMapPointEdgeStereo;
  vpMapPointEdgeStereo.reserve(nExpectedSize);

  const float thHuberMono = sqrt(5.991);
  const float thHuberStereo = sqrt(9.49);

  for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                 lend = lLocalMapPoints.end();
       lit != lend; lit++) {
    MapPoint* pMP = *lit;
    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
    int id = pMP->mnId + maxKFid + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    const map<KeyFrame*, size_t> observations = pMP->GetObservations();

    // Set edges
    for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(),
                                                mend = observations.end();
         mit != mend; mit++) {
      KeyFrame* pKFi = mit->first;
      if(pKFi->mbDRKF) continue;
      if (!pKFi->isBad()) {
        const cv::KeyPoint& kpUn = pKFi->mvKeysUn[mit->second];

        // Monocular observation
        if (pKFi->mvuRight[mit->second] < 0) {
          Eigen::Matrix<double, 2, 1> obs;
          obs << kpUn.pt.x, kpUn.pt.y;

          g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(pKFi->mnId)));
          e->setMeasurement(obs);
          const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
          e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuberMono);

          e->fx = pKFi->fx;
          e->fy = pKFi->fy;
          e->cx = pKFi->cx;
          e->cy = pKFi->cy;
          e->alpha = (pKFi->mDistCoef).at<float>(0);
          e->beta = (pKFi->mDistCoef).at<float>(1);

          optimizer.addEdge(e);
          vpEdgesMono.push_back(e);
          vpEdgeKFMono.push_back(pKFi);
          vpMapPointEdgeMono.push_back(pMP);
        } else  // Stereo observation
        {
          Eigen::Matrix<double, 4, 1> obs;
          const float kp_ur = pKFi->mvuRight[mit->second];
          const float kp_vr = pKFi->mvvRight[mit->second];
          obs << kpUn.pt.x, kpUn.pt.y, kp_ur,kp_vr;

          g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertex(pKFi->mnId)));
          e->setMeasurement(obs);
          const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
          Eigen::Matrix4d Info = Eigen::Matrix4d::Identity() * invSigma2;
          e->setInformation(Info);

          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuberStereo);

          e->fx = pKFi->fx;
          e->fy = pKFi->fy;
          e->cx = pKFi->cx;
          e->cy = pKFi->cy;
          e->alpha = (pKFi->mDistCoef).at<float>(0);
          e->beta = (pKFi->mDistCoef).at<float>(1);
          
          e->r_fx = pKFi->mpCamera2->mvParameters[0];
          e->r_fy = pKFi->mpCamera2->mvParameters[1];
          e->r_cx = pKFi->mpCamera2->mvParameters[2];
          e->r_cy = pKFi->mpCamera2->mvParameters[3];
          e->r_alpha = pKFi->mpCamera2->mvParameters[4];
          e->r_beta = pKFi->mpCamera2->mvParameters[5];
          e->R21 = R21;
          e->t2 = t2;

          optimizer.addEdge(e);
          vpEdgesStereo.push_back(e);
          vpEdgeKFStereo.push_back(pKFi);
          vpMapPointEdgeStereo.push_back(pMP);
        }
      }
    }
  }

  if (pbStopFlag)
    if (*pbStopFlag) return;

  optimizer.initializeOptimization();
  optimizer.optimize(5);

  bool bDoMore = true;

  if (pbStopFlag)
    if (*pbStopFlag) bDoMore = false;

  if (bDoMore) {
    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
      g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
      MapPoint* pMP = vpMapPointEdgeMono[i];

      if (pMP->isBad()) continue;

      if (e->chi2() > 5.991 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
      g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
      MapPoint* pMP = vpMapPointEdgeStereo[i];

      if (pMP->isBad()) continue;

      if (e->chi2() > 9.49 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    // Optimize again without the outliers
    //cout << "Start Optimize in [LM] " << __FUNCTION__ << endl;
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
  }

  vector<pair<KeyFrame*, MapPoint*> > vToErase;
  vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

  // Check inlier observations
  for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
    g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
    MapPoint* pMP = vpMapPointEdgeMono[i];

    if (pMP->isBad()) continue;

    if (e->chi2() > 5.991 || !e->isDepthPositive()) {
      KeyFrame* pKFi = vpEdgeKFMono[i];
      vToErase.push_back(make_pair(pKFi, pMP));
    }
  }

  for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
    MapPoint* pMP = vpMapPointEdgeStereo[i];

    if (pMP->isBad()) continue;

    if (e->chi2() > 9.49 || !e->isDepthPositive()) {
      KeyFrame* pKFi = vpEdgeKFStereo[i];
      vToErase.push_back(make_pair(pKFi, pMP));
    }
  }

  // Get Map Mutex
  unique_lock<mutex> lock(pMap->mMutexMapUpdate);

  if (!vToErase.empty()) {
    for (size_t i = 0; i < vToErase.size(); i++) {
      KeyFrame* pKFi = vToErase[i].first;
      MapPoint* pMPi = vToErase[i].second;
      pKFi->EraseMapPointMatch(pMPi);
      pMPi->EraseObservation(pKFi);
    }
  }

  // Recover optimized data

  // Keyframes
  for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                 lend = lLocalKeyFrames.end();
       lit != lend; lit++) {
    KeyFrame* pKF = *lit;
    g2o::VertexSE3Expmap* vSE3 =
        static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    pKF->SetPose(Converter::toCvMat(SE3quat));
  }

  // Points
  for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                 lend = lLocalMapPoints.end();
       lit != lend; lit++) {
    MapPoint* pMP = *lit;
    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(
        optimizer.vertex(pMP->mnId + maxKFid + 1));
    pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    pMP->UpdateNormalAndDepth();
  }
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag,
                                       Map* pMap/*, int& num_fixedKF only used for log in ORB_SLAM2*/)
{
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    //共视
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad()){
            lLocalKeyFrames.push_back(pKFi);
            }
    }

      // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                   lend = lLocalKeyFrames.end();
         lit != lend; lit++) {
      vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
      for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end();vit != vend; vit++) {
        MapPoint* pMP = *vit;
        if (pMP)
          if (!pMP->isBad())
            if (pMP->mnBALocalForKF != pKF->mnId) {
              lLocalMapPoints.push_back(pMP);
              pMP->mnBALocalForKF = pKF->mnId;
            }
      }
    }
    std::cout << "lLocalMapPoints = " << lLocalMapPoints.size() << std::endl;
  // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local
  // Keyframes
    list<KeyFrame*> lFixedCameras;
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                   lend = lLocalMapPoints.end();
         lit != lend; lit++) {
      map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
      for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                            mend = observations.end();
           mit != mend; mit++) {
        KeyFrame* pKFi = mit->first;
  
        if (pKFi->mnBALocalForKF != pKF->mnId &&
            pKFi->mnBAFixedForKF != pKF->mnId) {
          pKFi->mnBAFixedForKF = pKF->mnId;
          if (!pKFi->isBad())
            lFixedCameras.push_back(pKFi);
          else if (pKFi->isDRKF())  //||pKFi->IsScaled())  // DR!!! to fix scaled
                                    //KF after ResizeMap
            lFixedCameras.push_back(pKFi);
        }
      }
    }


    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mbInitKF==true);

        if (pKFi->isDRKF())  // temp for DR!!!
        {
          vSE3->setFixed(true);
          cout << __FUNCTION__ << " setFixed for KF: " << pKFi->mnId << endl;
        }
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<ORB_SLAM2::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM2::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);

    int nPoints = 0;

    int nKFs = lLocalKeyFrames.size()+lFixedCameras.size(), nEdges = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;
        //这里不同与3, GetObservations()返回的是能看到这个路标点的关键帧KeyFrame
        //,以及该路标点在对应关键帧中的id
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // Set edges
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
          KeyFrame* pKFi = mit->first;
          if(pKFi->mbDRKF) continue;
          //在3的基础上去除了对地图的判断
          if (!pKFi->isBad()) {
            const int leftIndex = (mit->second);

            // Monocular observation
            if (pKFi->mvuRight[(mit->second)] < 0) {
              const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
              Eigen::Matrix<double, 2, 1> obs;
              obs << kpUn.pt.x, kpUn.pt.y;

              ORB_SLAM2::EdgeSE3ProjectXYZ* e =
                  new ORB_SLAM2::EdgeSE3ProjectXYZ();

              e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                  optimizer.vertex(id)));
              e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                  optimizer.vertex(pKFi->mnId)));
              e->setMeasurement(obs);
              const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
              e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

              g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
              e->setRobustKernel(rk);
              rk->setDelta(thHuberMono);

              e->pCamera = pKFi->mpCamera;

              optimizer.addEdge(e);
              vpEdgesMono.push_back(e);
              vpEdgeKFMono.push_back(pKFi);
              vpMapPointEdgeMono.push_back(pMP);

              nEdges++;
            }
            //去除了是双目矫正后的优化边设置,因为不适用于这里

            if (pKFi->mpCamera2) {
              int leftIndex = (mit->second);
              cv::KeyPoint left_kp = pKFi->mvKeysUn[leftIndex];

              int rightIndex = pKFi->mvMatcheslr[leftIndex];//存放与左目点匹配的右目点的index
              if (rightIndex != -1) {
                Eigen::Matrix<double, 2, 1> obs;
                float right_u = pKFi->mvuRight[leftIndex];
                float right_v = pKFi->mvvRight[leftIndex];
                obs << right_u, right_v;

                ORB_SLAM2::EdgeSE3ProjectXYZToBody* e =
                    new ORB_SLAM2::EdgeSE3ProjectXYZToBody();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                    optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                    optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                //TODO: 这里先暂时使用左目的octave,但应该使用右目自己对应的层级
                const float& invSigma2 = pKFi->mvInvLevelSigma2[left_kp.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

                e->pCamera = pKFi->mpCamera2;

                optimizer.addEdge(e);
                vpEdgesBody.push_back(e);
                vpEdgeKFBody.push_back(pKFi);
                vpMapPointEdgeBody.push_back(pMP);

                nEdges++;
              }
            }
          }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    optimizer.optimize(5);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {
        // Check inlier observations
        int nMonoBadObs = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                nMonoBadObs++;
            }
        }

        int nBodyBadObs = 0;
        for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
        {
            ORB_SLAM2::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            MapPoint* pMP = vpMapPointEdgeBody[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                //e->setLevel(1);
                nBodyBadObs++;
            }

            //e->setRobustKernel(0);
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM2::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    //Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
    //相对于3, 去除了对于outliner的判断, 也就是与bRedrawError相关的内容

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        map<KeyFrame*, int> mspInitialConnectedKFs;
        map<KeyFrame*, int> mspInitialObservationKFs;

        //cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // Keyframes
    bool bShowStats = false;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                   lend = lLocalKeyFrames.end();
         lit != lend; lit++) {  //对所有的关键帧进行循环
      KeyFrame* pKFi = *lit;
      g2o::VertexSE3Expmap* vSE3 =
          static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
      g2o::SE3Quat SE3quat = vSE3->estimate();
      /*cv::Mat Tiw = Converter::toCvMat(SE3quat);
      cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
      cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
      double dist = cv::norm(trasl);*/
      pKFi->SetPose(Converter::toCvMat(SE3quat));  //设定优化后的新位姿

      /*if (dist > 1.0) {
        bShowStats = true;

        int numMonoMP = 0, numBadMonoMP = 0;
        int numStereoMP = 0, numBadStereoMP = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
          if (vpEdgeKFMono[i] != pKFi) continue;
          ORB_SLAM2::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
          MapPoint* pMP = vpMapPointEdgeMono[i];

          if (pMP->isBad()) continue;

          if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            numBadMonoMP++;
          } else {
            numMonoMP++;
          }
        }
      }*/
    }

    //Points恢复三维点的位姿
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::OptimizeEssentialGraph(
    Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const LoopClosing::KeyFrameAndPose& NonCorrectedSim3,
    const LoopClosing::KeyFrameAndPose& CorrectedSim3,
    const map<KeyFrame*, set<KeyFrame*> >& LoopConnections,
    const bool& bFixScale) {
  // Setup optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  g2o::BlockSolver_7_3::LinearSolverType* linearSolver =
      new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
  g2o::BlockSolver_7_3* solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

  solver->setUserLambdaInit(1e-16);
  optimizer.setAlgorithm(solver);

  const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
  const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

  const unsigned int nMaxKFid = pMap->GetMaxKFid();

  vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid + 1);
  vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(
      nMaxKFid + 1);
  vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

  const int minFeat = 100;

  // Set KeyFrame vertices
  for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;
    g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

    const int nIDi = pKF->mnId;

    LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

    if (it != CorrectedSim3.end()) {
      vScw[nIDi] = it->second;
      VSim3->setEstimate(it->second);
    } else {
      Eigen::Matrix<double, 3, 3> Rcw =
          Converter::toMatrix3d(pKF->GetRotation());
      Eigen::Matrix<double, 3, 1> tcw =
          Converter::toVector3d(pKF->GetTranslation());
      g2o::Sim3 Siw(Rcw, tcw, 1.0);
      vScw[nIDi] = Siw;
      VSim3->setEstimate(Siw);
    }

    if (pKF == pLoopKF) VSim3->setFixed(true);

    VSim3->setId(nIDi);
    VSim3->setMarginalized(false);
    VSim3->_fix_scale = bFixScale;

    optimizer.addVertex(VSim3);

    vpVertices[nIDi] = VSim3;
  }

  set<pair<long unsigned int, long unsigned int> > sInsertedEdges;

  const Eigen::Matrix<double, 7, 7> matLambda =
      Eigen::Matrix<double, 7, 7>::Identity();

  // Set Loop edges
  for (map<KeyFrame*, set<KeyFrame*> >::const_iterator
           mit = LoopConnections.begin(),
           mend = LoopConnections.end();
       mit != mend; mit++) {
    KeyFrame* pKF = mit->first;
    const long unsigned int nIDi = pKF->mnId;
    const set<KeyFrame*>& spConnections = mit->second;
    const g2o::Sim3 Siw = vScw[nIDi];
    const g2o::Sim3 Swi = Siw.inverse();

    for (set<KeyFrame*>::const_iterator sit = spConnections.begin(),
                                        send = spConnections.end();
         sit != send; sit++) {
      const long unsigned int nIDj = (*sit)->mnId;
      if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) &&
          pKF->GetWeight(*sit) < minFeat)
        continue;

      const g2o::Sim3 Sjw = vScw[nIDj];
      const g2o::Sim3 Sji = Sjw * Swi;

      g2o::EdgeSim3* e = new g2o::EdgeSim3();
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                          optimizer.vertex(nIDj)));
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                          optimizer.vertex(nIDi)));
      e->setMeasurement(Sji);

      e->information() = matLambda;

      optimizer.addEdge(e);

      sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
    }
  }

  // Set normal edges
  for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
    KeyFrame* pKF = vpKFs[i];

    const int nIDi = pKF->mnId;

    g2o::Sim3 Swi;

    LoopClosing::KeyFrameAndPose::const_iterator iti =
        NonCorrectedSim3.find(pKF);

    if (iti != NonCorrectedSim3.end())
      Swi = (iti->second).inverse();
    else
      Swi = vScw[nIDi].inverse();

    KeyFrame* pParentKF = pKF->GetParent();

    // Spanning tree edge
    if (pParentKF) {
      int nIDj = pParentKF->mnId;

      g2o::Sim3 Sjw;

      LoopClosing::KeyFrameAndPose::const_iterator itj =
          NonCorrectedSim3.find(pParentKF);

      if (itj != NonCorrectedSim3.end())
        Sjw = itj->second;
      else
        Sjw = vScw[nIDj];

      g2o::Sim3 Sji = Sjw * Swi;

      g2o::EdgeSim3* e = new g2o::EdgeSim3();
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                          optimizer.vertex(nIDj)));
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                          optimizer.vertex(nIDi)));
      e->setMeasurement(Sji);

      e->information() = matLambda;
      optimizer.addEdge(e);
    }

    // Loop edges
    const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
    for (set<KeyFrame*>::const_iterator sit = sLoopEdges.begin(),
                                        send = sLoopEdges.end();
         sit != send; sit++) {
      KeyFrame* pLKF = *sit;
      if (pLKF->mnId < pKF->mnId) {
        g2o::Sim3 Slw;

        LoopClosing::KeyFrameAndPose::const_iterator itl =
            NonCorrectedSim3.find(pLKF);

        if (itl != NonCorrectedSim3.end())
          Slw = itl->second;
        else
          Slw = vScw[pLKF->mnId];

        g2o::Sim3 Sli = Slw * Swi;
        g2o::EdgeSim3* el = new g2o::EdgeSim3();
        el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                             optimizer.vertex(pLKF->mnId)));
        el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                             optimizer.vertex(nIDi)));
        el->setMeasurement(Sli);
        el->information() = matLambda;
        optimizer.addEdge(el);
      }
    }

    // Covisibility graph edges
    const vector<KeyFrame*> vpConnectedKFs =
        pKF->GetCovisiblesByWeight(minFeat);
    for (vector<KeyFrame*>::const_iterator vit = vpConnectedKFs.begin();
         vit != vpConnectedKFs.end(); vit++) {
      KeyFrame* pKFn = *vit;
      if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) &&
          !sLoopEdges.count(pKFn)) {
        if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
          if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId),
                                             max(pKF->mnId, pKFn->mnId))))
            continue;

          g2o::Sim3 Snw;

          LoopClosing::KeyFrameAndPose::const_iterator itn =
              NonCorrectedSim3.find(pKFn);

          if (itn != NonCorrectedSim3.end())
            Snw = itn->second;
          else
            Snw = vScw[pKFn->mnId];

          g2o::Sim3 Sni = Snw * Swi;

          g2o::EdgeSim3* en = new g2o::EdgeSim3();
          en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                               optimizer.vertex(pKFn->mnId)));
          en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                               optimizer.vertex(nIDi)));
          en->setMeasurement(Sni);
          en->information() = matLambda;
          optimizer.addEdge(en);
        }
      }
    }
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(20);

  unique_lock<mutex> lock(pMap->mMutexMapUpdate);

  // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKFi = vpKFs[i];

    const int nIDi = pKFi->mnId;

    g2o::VertexSim3Expmap* VSim3 =
        static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
    g2o::Sim3 CorrectedSiw = VSim3->estimate();
    vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
    Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = CorrectedSiw.translation();
    double s = CorrectedSiw.scale();

    eigt *= (1. / s);  //[R t/s;0 1]

    cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

    pKFi->SetPose(Tiw);
  }

  // Correct points. Transform to "non-optimized" reference keyframe pose and
  // transform back with optimized pose
  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint* pMP = vpMPs[i];

    if (pMP->isBad()) continue;

    int nIDr;
    if (pMP->mnCorrectedByKF == pCurKF->mnId) {
      nIDr = pMP->mnCorrectedReference;
    } else {
      KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
      nIDr = pRefKF->mnId;
    }

    g2o::Sim3 Srw = vScw[nIDr];
    g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

    cv::Mat P3Dw = pMP->GetWorldPos();
    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw =
        correctedSwr.map(Srw.map(eigP3Dw));

    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
    pMP->SetWorldPos(cvCorrectedP3Dw);

    pMP->UpdateNormalAndDepth();
  }
}

int Optimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
                            vector<MapPoint*>& vpMatches1, g2o::Sim3& g2oS12,
                            const float th2, const bool bFixScale) {
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolverX::LinearSolverType* linearSolver;

  linearSolver =
      new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

  g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  // Calibration
  const cv::Mat& K1 = pKF1->mK;
  const cv::Mat& K2 = pKF2->mK;

  // Camera poses
  const cv::Mat R1w = pKF1->GetRotation();
  const cv::Mat t1w = pKF1->GetTranslation();
  const cv::Mat R2w = pKF2->GetRotation();
  const cv::Mat t2w = pKF2->GetTranslation();

  // Set Sim3 vertex
  g2o::VertexSim3Expmap* vSim3 = new g2o::VertexSim3Expmap();
  vSim3->_fix_scale = bFixScale;
  vSim3->setEstimate(g2oS12);
  vSim3->setId(0);
  vSim3->setFixed(false);
  vSim3->_principle_point1[0] = K1.at<float>(0, 2);
  vSim3->_principle_point1[1] = K1.at<float>(1, 2);
  vSim3->_focal_length1[0] = K1.at<float>(0, 0);
  vSim3->_focal_length1[1] = K1.at<float>(1, 1);
  vSim3->_principle_point2[0] = K2.at<float>(0, 2);
  vSim3->_principle_point2[1] = K2.at<float>(1, 2);
  vSim3->_focal_length2[0] = K2.at<float>(0, 0);
  vSim3->_focal_length2[1] = K2.at<float>(1, 1);
  optimizer.addVertex(vSim3);

  // Set MapPoint vertices
  const int N = vpMatches1.size();
  const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
  vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
  vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
  vector<size_t> vnIndexEdge;

  vnIndexEdge.reserve(2 * N);
  vpEdges12.reserve(2 * N);
  vpEdges21.reserve(2 * N);

  const float deltaHuber = sqrt(th2);

  int nCorrespondences = 0;

  for (int i = 0; i < N; i++) {
    if (!vpMatches1[i]) continue;

    MapPoint* pMP1 = vpMapPoints1[i];
    MapPoint* pMP2 = vpMatches1[i];

    const int id1 = 2 * i + 1;
    const int id2 = 2 * (i + 1);

    const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

    if (pMP1 && pMP2) {
      if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
        g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D1w = pMP1->GetWorldPos();
        cv::Mat P3D1c = R1w * P3D1w + t1w;
        vPoint1->setEstimate(Converter::toVector3d(P3D1c));
        vPoint1->setId(id1);
        vPoint1->setFixed(true);
        optimizer.addVertex(vPoint1);

        g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D2w = pMP2->GetWorldPos();
        cv::Mat P3D2c = R2w * P3D2w + t2w;
        vPoint2->setEstimate(Converter::toVector3d(P3D2c));
        vPoint2->setId(id2);
        vPoint2->setFixed(true);
        optimizer.addVertex(vPoint2);
      } else
        continue;
    } else
      continue;

    nCorrespondences++;

    // Set edge x1 = S12*X2
    Eigen::Matrix<double, 2, 1> obs1;
    const cv::KeyPoint& kpUn1 = pKF1->mvKeysUn[i];
    obs1 << kpUn1.pt.x, kpUn1.pt.y;

    g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
    e12->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
    e12->setVertex(
        1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e12->setMeasurement(obs1);
    const float& invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
    e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

    g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
    e12->setRobustKernel(rk1);
    rk1->setDelta(deltaHuber);
    e12->alpha = pKF1->mDistCoef.at<float>(0);
    e12->beta = pKF1->mDistCoef.at<float>(1);
    optimizer.addEdge(e12);

    // Set edge x2 = S21*X1
    Eigen::Matrix<double, 2, 1> obs2;
    const cv::KeyPoint& kpUn2 = pKF2->mvKeysUn[i2];
    obs2 << kpUn2.pt.x, kpUn2.pt.y;

    g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

    e21->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
    e21->setVertex(
        1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e21->setMeasurement(obs2);
    float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
    e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

    g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
    e21->setRobustKernel(rk2);
    rk2->setDelta(deltaHuber);
    e21->alpha = pKF2->mDistCoef.at<float>(0);
    e21->beta = pKF2->mDistCoef.at<float>(1);
    optimizer.addEdge(e21);

    vpEdges12.push_back(e12);
    vpEdges21.push_back(e21);
    vnIndexEdge.push_back(i);
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  // Check inliers
  int nBad = 0;
  for (size_t i = 0; i < vpEdges12.size(); i++) {
    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
    if (!e12 || !e21) continue;

    if (e12->chi2() > th2 || e21->chi2() > th2) {
      size_t idx = vnIndexEdge[i];
      vpMatches1[idx] = static_cast<MapPoint*>(NULL);
      optimizer.removeEdge(e12);
      optimizer.removeEdge(e21);
      vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
      vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
      nBad++;
    }
  }

  int nMoreIterations;
  if (nBad > 0)
    nMoreIterations = 10;
  else
    nMoreIterations = 5;

  if (nCorrespondences - nBad < 10) return 0;

  // Optimize again only with inliers

  optimizer.initializeOptimization();
  optimizer.optimize(nMoreIterations);

  int nIn = 0;
  for (size_t i = 0; i < vpEdges12.size(); i++) {
    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
    if (!e12 || !e21) continue;

    if (e12->chi2() > th2 || e21->chi2() > th2) {
      size_t idx = vnIndexEdge[i];
      vpMatches1[idx] = static_cast<MapPoint*>(NULL);
    } else
      nIn++;
  }

  // Recover optimized Sim3
  g2o::VertexSim3Expmap* vSim3_recov =
      static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
  g2oS12 = vSim3_recov->estimate();

  return nIn;
}

}  // namespace ORB_SLAM2