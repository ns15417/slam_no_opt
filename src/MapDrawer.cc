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

#include "MapDrawer.h"

#include "KeyFrame.h"
#include "MapPoint.h"

#define VISUAL

#ifdef VISUAL
#include <pangolin/pangolin.h>
#endif

#include <mutex>

namespace ORB_SLAM2 {

MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath) : mpMap(pMap) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
  mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
  mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
  mPointSize = fSettings["Viewer.PointSize"];
  mCameraSize = fSettings["Viewer.CameraSize"];
  mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
}

#ifdef VISUAL
void MapDrawer::DrawMapPoints() {
  const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
  const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

  set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

  if (vpMPs.empty()) return;

  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(0.0, 0.0, 0.0);

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])) continue;
    cv::Mat pos = vpMPs[i]->GetWorldPos();
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }
  glEnd();

  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(1.0, 0.0, 0.0);

  for (set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end();
       sit != send; sit++) {
    if ((*sit)->isBad()) continue;
    cv::Mat pos = (*sit)->GetWorldPos();
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }

  glEnd();
}

void MapDrawer::DrawGroundPlane() {
  // if(bDrawGroundPlane)
  {
    glLineWidth(2);
    glColor4f(0.1f, 0.1f, 0.1f, 0.2f);
    glBegin(GL_LINES);

    int nLines = 10;
    int width = 10;
    int size = 2;  // 5m

    for (int i = -nLines; i < nLines; i++) {
      glVertex3f(-width * size, 0, i * size);
      glVertex3f(width * size, 0, i * size);

      glVertex3f(i * size, 0, -width * size);
      glVertex3f(i * size, 0, width * size);
    }

    glEnd();
  }
}

void MapDrawer::DrawOdomPose(const bool bDrawOdomPose)
{
    const float &w=mKeyFrameSize;
    const float h=w*0.5;
    const float z=w*0.4;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if(bDrawOdomPose){
        for(size_t i=0; i<vpKFs.size();i++){
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twb = pKF->GetOdomPoseinWorld();
            cv::Mat Twb_t = Twb.t();

            glPushMatrix();
            glMultMatrixf(Twb_t.ptr<GLfloat>(0));
            glLineWidth(mKeyFrameLineWidth);
            glColor3f(1.0f,0.6f,0.0f);

            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(false)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);
        for(size_t i=0; i<vpKFs.size();i++){
            if(i==0) continue;
            cv::Mat Twb0 = vpKFs[i-1]->GetOdomPoseinWorld();
            cv::Mat Twb1 = vpKFs[i]->GetOdomPoseinWorld();

            cv::Mat b0 = Twb0.col(3).rowRange(0,3);
            cv::Mat b2 = Twb1.col(3).rowRange(0,3);
            glVertex3f(b0.at<float>(0),b0.at<float>(1),b0.at<float>(2));
            glVertex3f(b2.at<float>(0),b2.at<float>(1),b2.at<float>(2));
        }
        glEnd();
    }

}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph) {
  const float &w = mKeyFrameSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

  if (bDrawKF) {
    for (size_t i = 0; i < vpKFs.size(); i++) {
      KeyFrame *pKF = vpKFs[i];
      cv::Mat Twc = pKF->GetPoseInverse().t();

      glPushMatrix();

      glMultMatrixf(Twc.ptr<GLfloat>(0));

      glLineWidth(mKeyFrameLineWidth);
      if (pKF->isDRKF())
        glColor3f(1.0f, 0.0f, 0.0f);
      else if(pKF->IsScaled()) 
        glColor3f(0.0f, 0.0f, 0.0f);
      else if(pKF->mbInitKF)
        glColor3f(0.0f, 1.0f, 1.0f);
      else
        glColor3f(0.0f, 0.0f, 1.0f); 

      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(w, h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, h, z);

      glVertex3f(w, h, z);
      glVertex3f(w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(-w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);

      glVertex3f(-w, -h, z);
      glVertex3f(w, -h, z);
      glEnd();

      glPopMatrix();
    }
  }

  if (bDrawGraph) {
    glLineWidth(mGraphLineWidth);
    glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
    glBegin(GL_LINES);

    for (size_t i = 0; i < vpKFs.size(); i++) {
      // Covisibility Graph
      const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
      cv::Mat Ow = vpKFs[i]->GetCameraCenter();
      if (!vCovKFs.empty()) {
        for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(),
                                                vend = vCovKFs.end();
             vit != vend; vit++) {
          if ((*vit)->mnId < vpKFs[i]->mnId) continue;
          cv::Mat Ow2 = (*vit)->GetCameraCenter();
          glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
          glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
        }
      }

      // Spanning tree
      KeyFrame *pParent = vpKFs[i]->GetParent();
      if (pParent) {
        cv::Mat Owp = pParent->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
      }

      // Loops
      set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
      for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(),
                                     send = sLoopKFs.end();
           sit != send; sit++) {
        if ((*sit)->mnId < vpKFs[i]->mnId) continue;
        cv::Mat Owl = (*sit)->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
      }
    }

    glEnd();
  }
}

void MapDrawer::DrawRobot(const bool bDrawRobot) {
  const float &w = mKeyFrameSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

  if (bDrawRobot) {
    for (size_t i = 0; i < vpKFs.size(); i++) {
      KeyFrame *pKF = vpKFs[i];
      cv::Mat Twc = pKF->GetPoseInverse();
      cv::Mat RCw = pKF->GetRobotCenter();

      cv::Mat Tcw = pKF->GetPose();
      cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
      cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
      tcw.at<float>(2, 0) += 0.17;  // 0.1741;
      cv::Mat Rwc = Rcw.t();
      cv::Mat Ow = -Rwc * tcw;

      Twc = cv::Mat::eye(4, 4, Tcw.type());
      Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
      Ow.copyTo(Twc.rowRange(0, 3).col(3));

      // float x = RCw.at<float>(0,0);
      // float y = RCw.at<float>(1,0);
      // float z = RCw.at<float>(2,0);

      // Twc.at<float>(0,3) = x;
      // Twc.at<float>(1,3) = y;
      // Twc.at<float>(2,3) -= 1.0;

      cv::Mat Twc_t = Twc.t();

      // cout<<"[Map] x = "<< x << ", y = " << y << ", z = " << z <<endl;
      glPushMatrix();

      glMultMatrixf(Twc_t.ptr<GLfloat>(0));

      glLineWidth(mKeyFrameLineWidth);
      if (pKF->isDRKF())
        glColor3f(1.0f, 0.0f, 0.0f);
      else
        glColor3f(1.0f, 0.0f, 1.0f);

      glBegin(GL_LINES);

      glVertex3f(0, 0, 0);
      glVertex3f(w, h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, h, z);

      glVertex3f(w, h, z);
      glVertex3f(w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(-w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);

      glVertex3f(-w, -h, z);
      glVertex3f(w, -h, z);
      glEnd();

      glPopMatrix();
    }
  }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
  const float &w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();

#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(Twc.m);
#endif

  glLineWidth(mCameraLineWidth);
  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw) {
  unique_lock<mutex> lock(mMutexCamera);
  mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
  if (!mCameraPose.empty()) {
    cv::Mat Rwc(3, 3, CV_32F);
    cv::Mat twc(3, 1, CV_32F);
    {
      unique_lock<mutex> lock(mMutexCamera);
      Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
      twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
    }

    M.m[0] = Rwc.at<float>(0, 0);
    M.m[1] = Rwc.at<float>(1, 0);
    M.m[2] = Rwc.at<float>(2, 0);
    M.m[3] = 0.0;

    M.m[4] = Rwc.at<float>(0, 1);
    M.m[5] = Rwc.at<float>(1, 1);
    M.m[6] = Rwc.at<float>(2, 1);
    M.m[7] = 0.0;

    M.m[8] = Rwc.at<float>(0, 2);
    M.m[9] = Rwc.at<float>(1, 2);
    M.m[10] = Rwc.at<float>(2, 2);
    M.m[11] = 0.0;

    M.m[12] = twc.at<float>(0);
    M.m[13] = twc.at<float>(1);
    M.m[14] = twc.at<float>(2);
    M.m[15] = 1.0;
  } else
    M.SetIdentity();
}

#endif

}  // namespace ORB_SLAM2
