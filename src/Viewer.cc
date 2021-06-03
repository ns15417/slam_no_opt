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

#include "Viewer.h"
#define VISUAL

#ifdef VISUAL
#include <pangolin/pangolin.h>
#endif
#include <mutex>

extern bool g_pause;

class DRVisualizer {
  cv::Mat plot;
  float last_pose_x;
  float last_pose_y;
  float scale;
  int width;
  int height;

 public:
  void init(int px, int py, int w, int h, float s) {
    width = w;
    height = h;
    scale = s;

    // plot.create(h, w, CV_8UC3);
    plot = cv::Mat::zeros(h, w, CV_8UC3);

    last_pose_x = 0;
    last_pose_y = 0;

    clear();

    int nLine = 30;

    // 5m
    int size = 5 * s;
    for (int i = -nLine; i < nLine; i++) {
      cv::Point pt1, pt2;
      pt1.x = 0;
      pt2.x = w;
      pt1.y = i * size;
      pt2.y = i * size;
      cv::line(plot, cv::Point(0, h / 2 + i * size),
               cv::Point(w, h / 2 + i * size), cv::Scalar(200, 200, 200));
      cv::line(plot, cv::Point(w / 2 + i * size, 0),
               cv::Point(w / 2 + i * size, w), cv::Scalar(200, 200, 200));
    }
  };

  void visualize(float x, float y) {
    // cout<<"x ="<<x<<", y="<<y<<endl;

    cv::Point pt1, pt2;
    pt1.y = -last_pose_x*scale + height/2;
    pt1.x = -last_pose_y*scale + width/2;
 
    pt2.y = -x*scale + height/2;
    pt2.x = -y*scale + width/2;

    // cout<<"pt1="<<pt1<<", pt2="<<pt2<<endl;
    cv::line(plot, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    last_pose_x = x;
    last_pose_y = y;
  };

  void clear() { plot = cv::Scalar(255, 255, 255); };
};

namespace ORB_SLAM2 {

Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer,
               MapDrawer *pMapDrawer, Tracking *pTracking,
               const string &strSettingPath)
    : mpSystem(pSystem),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpTracker(pTracking),
      mbFinishRequested(false),
      mbFinished(true),
      mbStopped(true),
      mbStopRequested(false) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  float fps = fSettings["Camera.fps"];
  if (fps < 1) fps = 30;
  mT = 1e3 / fps;

  mImageWidth = fSettings["Camera.width"];
  mImageHeight = fSettings["Camera.height"];
  if (mImageWidth < 1 || mImageHeight < 1) {
    mImageWidth = 640;
    mImageHeight = 480;
  }

  mViewpointX = fSettings["Viewer.ViewpointX"];
  mViewpointY = fSettings["Viewer.ViewpointY"];
  mViewpointZ = fSettings["Viewer.ViewpointZ"];
  mViewpointF = fSettings["Viewer.ViewpointF"];

  mViewDRPPM = fSettings["Viewer.DRPixelPerMeter"];
  mPosRGBX = fSettings["Viewer.PosRGBX"];
  mPosRGBY = fSettings["Viewer.PosRGBY"];
  mPosDepthX = fSettings["Viewer.PosDepthX"];
  mPosDepthY = fSettings["Viewer.PosDepthY"];
  mPosDRX = fSettings["Viewer.PosDRX"];
  mPosDRY = fSettings["Viewer.PosDRY"];
  mPosMapViewerX = fSettings["Viewer.PosMapViewerX"];
  mPosMapViewerY = fSettings["Viewer.PosMapViewerY"];
}
#ifdef VISUAL
void Viewer::Run() {
  mbFinished = false;
  mbStopped = false;

  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Issue specific OpenGl we might need
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
  pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
  pangolin::Var<bool> menuShowOdomPose("menu.ShowOdom",true,true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
  pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false,
                                           true);
  // pangolin::Var<bool> menuLocalizationMode("menu.Localization
  // Mode",false,true);
  pangolin::Var<bool> menuSaveMap("menu.Save Map", false, true);

  pangolin::Var<bool> menuReset("menu.Reset", false, false);
  pangolin::Var<bool> menuShutDown("menu.Shut Down", false, false);
  pangolin::Var<bool> menuPauseResume("menu.Pause/Resume", false, false);
  pangolin::Var<bool> menuStop("menu.Stop", false, false);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389,
                                 0.1, 1000),
      pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0,
                                0.0, -1.0, 0.0));

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                         1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  cv::namedWindow("ORB-SLAM2: Current Frame");

  bool bFollow = true;
  bool bLocalizationMode = false;

  //DRVisualizer dr_vis;
  //dr_vis.init(mPosDRX, mPosDRY, 640, 480, mViewDRPPM);

  while (1) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

    if (menuFollowCamera && bFollow) {
      s_cam.Follow(Twc);
    } else if (menuFollowCamera && !bFollow) {
      s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
          mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
      s_cam.Follow(Twc);
      bFollow = true;
    } else if (!menuFollowCamera && bFollow) {
      bFollow = false;
      s_cam.SetModelViewMatrix(
          pangolin::ModelViewLookAt(0, -10, 0, 0, 0, 0, 0.0, 0.0, 1.0));

      pangolin::OpenGlMatrix Twc1;
      Twc1.m[0] = 1;
      Twc1.m[1] = 0;
      Twc1.m[2] = 0;
      Twc1.m[3] = Twc1.m[3];
      Twc1.m[4] = 0;
      Twc1.m[5] = 1;
      Twc1.m[6] = 0;
      Twc1.m[7] = Twc1.m[7];
      Twc1.m[8] = 0;
      Twc1.m[9] = 0;
      Twc1.m[10] = 1;
      Twc1.m[11] = Twc1.m[11];
      s_cam.Follow(Twc1);
    }

    if (menuLocalizationMode && !bLocalizationMode) {
      mpSystem->ActivateLocalizationMode();
      bLocalizationMode = true;
    } else if (!menuLocalizationMode && bLocalizationMode) {
      mpSystem->DeactivateLocalizationMode();
      bLocalizationMode = false;
    }

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    mpMapDrawer->DrawCurrentCamera(Twc);
    if (menuShowKeyFrames || menuShowGraph)
      mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
    if (menuShowPoints) mpMapDrawer->DrawMapPoints();
    if(menuShowOdomPose) mpMapDrawer->DrawOdomPose(menuShowOdomPose);
    // mpMapDrawer->DrawRobot(true);
    mpMapDrawer->DrawGroundPlane();

    pangolin::FinishFrame();

    cv::Mat im = mpFrameDrawer->DrawFrame();
    cv::imshow("ORB-SLAM2: Current Frame", im);
    // mId = mpFrameDrawer->mnId;
    // std::cout<<"ID: "<<mId<<endl;
    // if(mId == 530)
    // cv::imwrite("/home/leo/dataset/result/EUCM1.jpg", im);

    // cout<<"DR_x="<<mpTracker->DR_x<<",DR_y="<<mpTracker->DR_y<<endl;
    //dr_vis.visualize(mpTracker->DR_x, mpTracker->DR_y);

    cv::waitKey(mT);

    if (menuReset) {
      menuShowGraph = true;
      menuShowKeyFrames = true;
      menuShowPoints = true;
      menuLocalizationMode = false;
      if (bLocalizationMode) mpSystem->DeactivateLocalizationMode();
      bLocalizationMode = false;
      bFollow = true;
      menuFollowCamera = true;
      mpSystem->Reset();
      menuReset = false;
    }

    if (menuPauseResume) {
      g_pause = !g_pause;
      menuPauseResume = false;
    }

    if (menuStop) mpSystem->Stop();

    if (Stop()) {
      while (isStopped()) {
        usleep(3000);
      }
    }

    if (CheckFinish()) break;
  }

  SetFinish();
}
#endif
void Viewer::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool Viewer::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void Viewer::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
}

bool Viewer::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

void Viewer::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  if (!mbStopped) mbStopRequested = true;
}

bool Viewer::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

bool Viewer::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);

  if (mbFinishRequested)
    return false;
  else if (mbStopRequested) {
    mbStopped = true;
    mbStopRequested = false;
    return true;
  }

  return false;
}

void Viewer::Release() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopped = false;
}

}  // namespace ORB_SLAM2
