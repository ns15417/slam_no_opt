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

#include <cv_bridge/cv_bridge.h>
#include <math.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <stdlib.h>
#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "../../../include/System.h"
#include "../../../include/Converter.h"
#include "eigen3/Eigen/Dense"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/TransformStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "tf2_ros/transform_listener.h"

using namespace std;
using namespace ORB_SLAM2;
//using namespace fisheye_orb_ros;
#define TF
#define COMPRESS
bool g_pause = false;
typedef struct {
  float x;
  float y;
  float th;
} POSE2D;
POSE2D cur_pose;
std::mutex m;

POSE2D first_pose;
bool smallID(const KeyFrame *k1, const KeyFrame *k2) {
  return k1->mnId < k2->mnId;
}
cv::Size image_size(640, 480);
cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
cv::Mat mapy = cv::Mat(image_size, CV_32FC1);

class ImageGrabber {
 public:
  ImageGrabber(ORB_SLAM2::System *pSLAM, const std::string filepath);
  void GrabOdom(const geometry_msgs::TransformStamped transform);
#ifdef COMPRESS
  void GrabStereo(const sensor_msgs::CompressedImageConstPtr& msgLeft, const sensor_msgs::CompressedImageConstPtr& msgRight);
#else
  void GrabStereo(const sensor_msgs::ImageConstPtr & msgLeft, const sensor_msgs::ImageConstPtr &msgRight);
#endif

  int PublishTF(cv::Mat &Tcw_);
  int PublishPointCloud(cv::Mat &Tcw_);
  void PublishTrajectory();
  int OrbSlamUpdate(cv::Mat &left_img, cv::Mat &right_img, std::string timestamp_str);
  void SetCam2Baselink(cv::Mat &Tbc);
  // bool getDetection(bbox_to_orb::Request &req,bbox_to_orb::Response &res);
  void PubStereoptClouds();
#ifdef PCL
  int PCAPlanFitting(pcl::PointCloud<pcl::PointXYZ>::Ptr  inlierPoints);
  int PubFilteredPoints();
  void SaveFilterPoints(pcl::PointCloud<pcl::PointXYZ> finalpoints);
#endif
  nav_msgs::Path ORB_path;
  pcl::PointCloud<pcl::PointXYZ> pcl_Mapcloud;
  sensor_msgs::PointCloud2 ROSMSG_ptcloud;
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

  ros::NodeHandle nodeHandler;
  ros::Publisher PointCloud_pub =
      nodeHandler.advertise<sensor_msgs::PointCloud2>("CurFrame_Points", 1);
  ros::Publisher path_pub =
      nodeHandler.advertise<nav_msgs::Path>("keyframe_path", 1, true);
  geometry_msgs::TransformStamped geotransform;
  tf::Transform cur_tf_pose;
  tf::TransformBroadcaster broadcaster;

  // c:cam b:baselink
  cv::Mat Tbc = cv::Mat::eye(4, 4, CV_32F);
  std::vector<POSE2D> odom_poses;
  std::vector<double> odom_times;
  ORB_SLAM2::System *mpSLAM;
  std::string setting_filepath;
  std::string rootPath;
  int frameid = 0;
  int dr_idx = 0;
  POSE2D old_pose = {0, 0, 0};

  float g_del_x = 0;
  float g_del_y = 0;
  float g_del_th = 0;

  int finishedCurTrack = 1;
  int temp_num = 0;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "Stereo");
  ros::start();

  if (argc != 8) {
    cerr << endl
         << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings "
            "SAVEorLOAD rootPath map_name left_topic right_topic"
         << endl;
    ros::shutdown();
    return 1;
  }

  ORB_SLAM2::System MYSLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, true);
  ImageGrabber igb(&MYSLAM, argv[2]);
  std::string mapStatus = argv[3];
  std::string rootpath = argv[4];
  std::cout << "rootpath: " << rootpath << endl;
  std::string mapname = argv[5];
  if (mapStatus == "LOAD") {MYSLAM.LoadMap(mapname);}

  igb.SetCam2Baselink(igb.Tbc);
  igb.ORB_path.header.stamp = ros::Time::now();
  igb.ORB_path.header.frame_id = "map";
  igb.rootPath = rootpath;
  
  std::string left_topic = argv[6];
  std::string right_topic = argv[7];
  ros::NodeHandle nodeHandler;
  tf2_ros::Buffer tf_buffer{::ros::Duration(1)};
  tf2_ros::TransformListener listener(tf_buffer, nodeHandler, true);
  //ros::ServiceServer service = nodeHandler.advertiseService("bbox_to_orb",&ImageGrabber::getDetection,&igb);
#ifdef COMPRESS
  message_filters::Subscriber<sensor_msgs::CompressedImage> left_sub(nodeHandler, left_topic, 1);
  message_filters::Subscriber<sensor_msgs::CompressedImage> right_sub(nodeHandler, right_topic, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));
#else
// 接收非压缩左右目图像
  message_filters::Subscriber<sensor_msgs::Image> left_sub(nodeHandler, left_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(nodeHandler, right_topic, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));
#endif
  uint64_t oldtimestamp = 0;
  while (ros::ok()) {
  #ifdef TF
    try {
      igb.temp_num++;
      igb.geotransform =tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
      uint64_t timestamp = igb.geotransform.header.stamp.toNSec();
      if (oldtimestamp != timestamp) {
        oldtimestamp = timestamp;
        igb.GrabOdom(igb.geotransform);
      }
    } catch (tf::TransformException &ex) {
      ROS_ERROR("%s", ex.what());
    }
    usleep(2e5);
    if (igb.temp_num > 120) {
      break;
    }
  #endif
    ros::spinOnce();
  }

  MYSLAM.Shutdown();
  cout << "Go to save Trajectory and map" << endl;
  if(mapStatus == "SAVE"){
    MYSLAM.SaveKeyFrameTrajectoryTUM(rootpath + "KeyFrameTrajectory.txt");
    std::string savedmap_name = mapname+"_map.bin";
    MYSLAM.SaveMap(savedmap_name);
    cout << "    Saved Map and KeyFrame to " << savedmap_name << endl;
  }
  cout << "Finish shutdown" << endl;
  ros::shutdown();
  return 0;
}

ImageGrabber::ImageGrabber(ORB_SLAM2::System *pSLAM, const std::string filepath)
    : mpSLAM(pSLAM), setting_filepath(filepath) {
  cv::FileStorage fsSettings(filepath.c_str(), cv::FileStorage::READ);
}

void ImageGrabber::GrabOdom(const geometry_msgs::TransformStamped transform) {
  tf::transformMsgToTF(transform.transform, cur_tf_pose);
  double odom_time = transform.header.stamp.toNSec();
  float x = cur_tf_pose.getOrigin().x();
  float y = cur_tf_pose.getOrigin().y();
  double roll, pitch, yaw;
  tf::Matrix3x3(cur_tf_pose.getRotation()).getRPY(roll, pitch, yaw);
  POSE2D odom_pose;
  odom_pose.x = x;
  odom_pose.y = y;
  odom_pose.th = yaw;
  odom_poses.push_back(odom_pose);
  odom_times.push_back(odom_time);
}


#ifdef COMPRESS
void ImageGrabber::GrabStereo(
  const sensor_msgs::CompressedImageConstPtr& msgLeft,
  const sensor_msgs::CompressedImageConstPtr& msgRight) {

  cv::Mat left_img;
  try {
    left_img = cv::imdecode(cv::Mat(msgLeft->data), 0);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat right_img;
  try {
    right_img = cv::imdecode(cv::Mat(msgRight->data), 0);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (!left_img.empty() && !right_img.empty()&&finishedCurTrack) {
    temp_num = 0;
    finishedCurTrack = 0;
    //clahe->apply(left_img,left_img);
    //clahe->apply(right_img,right_img);
    std::string ts_str = std::to_string(msgLeft->header.stamp.toNSec());
    cout << "图像话题读取成功！开始 OrbSlamUpdate函数 ！" << ts_str << endl;
    OrbSlamUpdate(left_img, right_img, ts_str);
    finishedCurTrack = 1;
  }
}
#else
void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr & msgLeft, const sensor_msgs::ImageConstPtr &msgRight){
  cout << "get image data" << endl;
  cv::Mat leftimg,rightimg;
  cv_bridge::CvImageConstPtr cv_ptr_left;
  cv_bridge::CvImageConstPtr cv_ptr_right;
  try {
    cv_ptr_left = cv_bridge::toCvShare(msgLeft);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("Could not convert to image! ");
  }

  try {
    cv_ptr_right = cv_bridge::toCvShare(msgRight);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("Could not convert to image! ");
  }


  leftimg = cv_ptr_left->image;
  rightimg = cv_ptr_right->image;

  if (!leftimg.empty() && !rightimg.empty()&&finishedCurTrack) {
    temp_num = 0;
    finishedCurTrack = 0;
    std::string ts_str = std::to_string(cv_ptr_left->header.stamp.toNSec());
    cout << "图像话题读取成功！开始 OrbSlamUpdate函数 ！" << ts_str << endl;
    OrbSlamUpdate(leftimg, rightimg, ts_str);
    finishedCurTrack = 1;
  }
}
#endif

int ImageGrabber::OrbSlamUpdate(cv::Mat &left_img,cv::Mat &right_img, std::string timestamp_str) {
  double timestamp = std::stod(timestamp_str);
  cout << "[OrbSlamUpdate]: " << setprecision(18) << timestamp <<endl;
#ifdef TF
  if (odom_poses.empty()) {
    // return 0; //这里可能是造成图像卡顿的原因，因为一直在等待odom的数据
  }
  std::vector<POSE2D> tmp_odom_poses;
  std::vector<double> tmp_odom_times;
  {
    std::unique_lock<std::mutex> q(m);
    tmp_odom_poses = odom_poses;
    tmp_odom_times = odom_times;
    if (odom_poses.size() > 2) {
      odom_poses.erase(odom_poses.begin(), odom_poses.end() - 1);
      odom_times.erase(odom_times.begin(), odom_times.end() - 1);
    }
  }

  if (frameid == 0 && tmp_odom_poses.size() < 2) {
    cout << " No odom data for this image then continue " << tmp_odom_poses.size() << endl;
    //  No odom data for this image then continue 0
    return -1;
  } else if (tmp_odom_poses.size() < 2) {
    cout << "odom_poses size : " << tmp_odom_poses.size() << endl;
    mpSLAM->SetOdomFlag(false);
  }

  if (frameid == 0 && tmp_odom_poses.size() != 0) {
    old_pose = tmp_odom_poses[tmp_odom_poses.size() - 1];
    mpSLAM->SetFirstDRpose(old_pose.x, old_pose.y, old_pose.th);
    first_pose.x = old_pose.x;
    first_pose.y = old_pose.y;
    first_pose.th = old_pose.th;
    frameid++;
  }
  int id = 0;

  while (tmp_odom_times[id] < timestamp) {
    cur_pose = tmp_odom_poses[id];
    id++;
    if (id == tmp_odom_times.size()) {
      break;
    }
  }
  mpSLAM->SetOdomFlag(true);
  g_del_x = cur_pose.x - old_pose.x;
  g_del_y = cur_pose.y - old_pose.y;
  g_del_th = cur_pose.th - old_pose.th;
  mpSLAM->SetDR(cur_pose.x, cur_pose.y, cur_pose.th, g_del_x, g_del_y, g_del_th);
  old_pose = cur_pose;

  // cout << "g_del_th " << g_del_th << endl;
  // std::cout << "cur_pose pose : " << cur_pose.x << " " << cur_pose.y << "  " << cur_pose.th << endl;
  // std::cout << "current theta compared to " << (cur_pose.th - first_pose.th) / M_PI * 180.0 << endl;
#endif

  cv::Mat new_left_img = left_img.clone();
  cv::Mat new_right_img = right_img.clone();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  cv::Mat ret_Tcw = mpSLAM->TrackStereo(left_img, right_img, timestamp);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  cout << "----------------Cost : " << setprecision(18) << ttrack << " --------------" << endl;

  frameid++;
  if (!ret_Tcw.empty()) {
  }
  return 1;
}

int ImageGrabber::PublishTF(cv::Mat &Tcw_) {
  tf::Transform transform;
  cv::Mat Tcw;
  Tcw_.copyTo(Tcw);
  cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
  cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
  cv::Mat Twc = cv::Mat::eye(4, 4, Tcw.type());

  Rwc.copyTo(Twc.colRange(0, 3).rowRange(0, 3));
  twc.copyTo(Twc.rowRange(0, 3).col(3));

  cv::Mat Tmb;  // baselink to map, map is considered as the original baselink
  Tmb = Tbc * Twc;  //*Tbc.inv(); //temp test: cam in origin baselink
  transform.setOrigin(tf::Vector3(Tmb.at<float>(0, 3), Tmb.at<float>(1, 3),
                                  Tmb.at<float>(2, 3)));

  tf::Matrix3x3 r;
  r.setValue(Tmb.at<float>(0, 0), Tmb.at<float>(0, 1), Tmb.at<float>(0, 2),
             Tmb.at<float>(1, 0), Tmb.at<float>(1, 1), Tmb.at<float>(1, 2),
             Tmb.at<float>(2, 0), Tmb.at<float>(2, 1), Tmb.at<float>(2, 2));
  transform.setBasis(r);
  broadcaster.sendTransform(tf::StampedTransform(
      tf::Transform(transform), ros::Time::now(), "map", "ORB_BASELINK"));
  return 1;
}

void ImageGrabber::PublishTrajectory() {
  ORB_path.poses.clear();
  std::vector<KeyFrame *> allKeyframes = mpSLAM->GellAllKeyFrames();
  sort(allKeyframes.begin(), allKeyframes.end(), smallID);  // 按照ID升序排序；
  geometry_msgs::PoseStamped pose;
  for (int i = 0; i < allKeyframes.size(); i++) {
    cv::Mat curPose_Twc =
        allKeyframes[i]->GetPoseInverse();          // cam to camera map
    cv::Mat Tbw_c = Tbc * curPose_Twc;              // camera to odom map
    cv::Mat tbinmap = Tbw_c.col(3).rowRange(0, 3);  // camera pose in odom map

    // cv::Mat Twb = Tbw.inv();
    // cv::Mat Tmb= Tbc*Twb;
    // cv::Mat tbinmap = Tmb.col(3).rowRange(0,3);
    // cv::Mat tbinmap = Twb.col(3).rowRange(0,3);

    pose.pose.position.x = tbinmap.at<float>(0);
    pose.pose.position.y = tbinmap.at<float>(1);
    pose.pose.position.z = 0;
    pose.pose.orientation.x = 0;
    pose.pose.orientation.y = 0;
    pose.pose.orientation.z = 0;
    pose.pose.orientation.w = 1;

    ORB_path.poses.push_back(pose);
  }
}

int ImageGrabber::PublishPointCloud(cv::Mat &Tcw_) {
  cv::Mat curTcw;
  Tcw_.copyTo(curTcw);
  std::vector<MapPoint *> allMapPoints = mpSLAM->GetAllPointsinMap();
  int validpts = 0;
  int pointsize = allMapPoints.size();
  for (auto mappoint : allMapPoints)
    if (mappoint != NULL) validpts++;
  pcl_Mapcloud.width = validpts;
  pcl_Mapcloud.height = 1;
  pcl_Mapcloud.points.resize(pcl_Mapcloud.width * pcl_Mapcloud.height);

  validpts = 0;
  for (int i = 0; i < pointsize; i++) {
    auto curmappt = allMapPoints[i];
    if (curmappt == NULL || curmappt->GetWorldPos().at<float>(1) < -1.0 ||
        curmappt->GetWorldPos().at<float>(1) > 0.2)
      continue;

    cv::Mat WorldPos = cv::Mat_<float>(4, 1);  // points pose in ORB_worldFrame
    WorldPos.at<float>(0) = curmappt->GetWorldPos().at<float>(0);
    WorldPos.at<float>(1) = curmappt->GetWorldPos().at<float>(1);
    WorldPos.at<float>(2) = curmappt->GetWorldPos().at<float>(2);
    WorldPos.at<float>(3) = 1;
    // points in baselin worldframe,actually is map
    cv::Mat poi = Tbc * WorldPos;
    pcl_Mapcloud.points[validpts].x = poi.at<float>(0);
    pcl_Mapcloud.points[validpts].y = poi.at<float>(1);
    pcl_Mapcloud.points[validpts].z = poi.at<float>(2);

    validpts++;
  }
}

void ImageGrabber::SetCam2Baselink(cv::Mat &Tbc) {
  cv::Mat Rbc(3, 3, CV_32F);
  Rbc.at<float>(0, 0) = 0.0;
  Rbc.at<float>(0, 1) = 0.0;
  Rbc.at<float>(0, 2) = 1.0;
  Rbc.at<float>(1, 0) = -1.0;
  Rbc.at<float>(1, 1) = 0.0;
  Rbc.at<float>(1, 2) = 0.0;
  Rbc.at<float>(2, 0) = 0.0;
  Rbc.at<float>(2, 1) = -1.0;
  Rbc.at<float>(2, 2) = 0.0;
  cv::Mat tbc(3, 1, CV_32F);
  // tbc.at<float>(0) = 0; tbc.at<float>(1) = 0; tbc.at<float>(2)= 0;
  tbc.at<float>(0) = 0.1162;
  tbc.at<float>(1) = 0;
  tbc.at<float>(2) = 0.2355;
  // tbc.at<float>(0) = 0.1162; tbc.at<float>(1) = 0.05368174; tbc.at<float>(2)=
  // 0.3091;

  cv::Mat add_T = cv::Mat::eye(4, 4, CV_32F);
  Eigen::Vector3d angles(0, 0, 0.054359878);
  // Eigen::Vector3d angles(0,0,0.02);
  Eigen::Matrix<double, 3, 3> Rtest;
  Rtest = Converter::angletocvMat(angles);
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++) {
      add_T.at<float>(row, col) = Rtest(row, col);
    }
  cv::Mat ori_Tbc = cv::Mat::eye(4, 4, CV_32F);
  Rbc.copyTo(ori_Tbc.colRange(0, 3).rowRange(0, 3));
  tbc.copyTo(ori_Tbc.rowRange(0, 3).col(3));
  Tbc = add_T * ori_Tbc;
}

/*
bool ImageGrabber::getDetection(bbox_to_orb::Request &req,
                                bbox_to_orb::Response &res) {
  std::vector<cv::Rect> vbbox;
  std::string req_timestamp;
  std::vector<std::vector<cv::Point3f>> res_points;
  std::vector<bbox_msg> req_list = req.bbox_list;
  std::string KF_time = req.timestamp;
  cout << "[Detection] asked from timestamp " << setprecision(18) << KF_time << endl;
  cout << "[Detection] detected " << req_list.size() << " bounding box " << endl;
  for (int i = 0; i < req_list.size(); i++) {
    bbox_msg cur_box = req_list[i];
    cv::Point2f tl, br;
    tl.x = cur_box.top_left.x;
    tl.y = cur_box.top_left.y;
    br.x = cur_box.bottom_right.x;
    br.y = cur_box.bottom_right.y;
    cv::Rect mybbox(tl, br);
    vbbox.push_back(mybbox);
  }
  req_timestamp = KF_time;
  
  mpSLAM->GetMapPointsinbbox(req_timestamp, vbbox, res_points);

  std::vector<orb_point_msg> res_pts;
  for (int i = 0; i < res_points.size(); i++) {
    std::vector<geometry_msgs::Point32> ros_pts;
    for (int j = 0; j < res_points[i].size(); j++) {
      cv::Point3f tmp_point = res_points[i][j];
      geometry_msgs::Point32 ros_pt;
      ros_pt.x = tmp_point.x;
      ros_pt.y = tmp_point.y;
      ros_pt.z = tmp_point.z;
      ros_pts.push_back(ros_pt);
    }
    orb_point_msg res_pt;
    res_pt.orb_points = ros_pts;
    res_pts.push_back(res_pt);
    cout << "[Detection] respond " << res_points[i].size() << " mappoints for bounding box " << i << endl;
  }
  
  res.orb_point_list = res_pts;
  return true;
}*/