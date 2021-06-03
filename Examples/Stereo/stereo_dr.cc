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

#include <System.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <opencv2/core/core.hpp>

#define DR
#define Skip
using namespace std;
typedef struct {
  float x;
  float y;
  float th;
} POSE2D;

bool g_pause = false;
void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimeStamps);
void readOdomData(string fname, vector<POSE2D>& pose_data,
                  vector<double>& pose_times);

void readSkipData(string fname, vector<int>& track_fail_start,
                  vector<int>& track_fail_end);

int main(int argc, char **argv) {
  if (argc != 8) {
    cerr << endl
         << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings "
            "root_path path_to_ass_file dataset_name map_status startframeid"
         << endl;
    return 1;
  }

  // Retrieve paths to images
  vector<string> vstrImageLeft;
  vector<string> vstrImageRight;
  vector<double> vTimeStamps;
  LoadImages(string(argv[3]), string(argv[3]), string(argv[4]), vstrImageLeft,
             vstrImageRight, vTimeStamps);

  if (vstrImageLeft.empty() || vstrImageRight.empty()) {
    cerr << "ERROR: No images in provided path." << endl;
    return 1;
  }

  if (vstrImageLeft.size() != vstrImageRight.size()) {
    cerr << "ERROR: Different number of left and right images." << endl;
    return 1;
  }

  const int nImages = vstrImageLeft.size();

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, true);

  std::string dataset_name = argv[5];
  std::string map_name =
      "/home/shinan/Project/FisheyeSLAM-DR-SE-master/" + dataset_name + ".bin";
  std::string mapStatus = argv[6];
  if(mapStatus == "LOAD") SLAM.LoadMap(map_name);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);
  int startframeid = std::stoi(argv[7]);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

#ifdef DR
  vector<POSE2D> odom_poses;
  vector<double> odom_times;
  string odom_file(argv[3]);
  odom_file += "/l-slam.txt";
  cout << odom_file << endl;
  readOdomData(odom_file, odom_poses, odom_times);

  POSE2D old_pose = {0, 0, 0};
  POSE2D cur_pose;
  float g_del_x = 0;
  float g_del_y = 0;
  float g_del_th = 0;

  int dr_idx = 0;
  while (odom_times[dr_idx] < (vTimeStamps[startframeid])) {
    dr_idx++;
  }
  dr_idx--;

  old_pose = odom_poses[dr_idx];
  int frame_id = 0;
#endif

  // simulate tracking fail
  vector<int> track_fail_start;
  vector<int> track_fail_end;

  string f_skipdata(argv[3]);
  f_skipdata += "/skip.txt";
  readSkipData(f_skipdata, track_fail_start, track_fail_end);

  // Main loop
  cv::Mat imLeft, imRight, imLeftRect, imRightRect;
  for (int ni = startframeid; ni < nImages; ni++) {
    // Read left and right images from file
    imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
    imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

    if (imLeft.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageLeft[ni]) << endl;
      return 1;
    }else
    {
      cout << " >>>>>> Reading left image: " << vstrImageLeft[ni] << std::endl;
    }
    

    if (imRight.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageRight[ni]) << endl;
      return 1;
    }else
    {
      cout << " >>>>>> Reading left image: " << vstrImageRight[ni] << std::endl;
    }
    
#ifdef Skip
    for (int j = 0; j < track_fail_start.size(); j++)  // simulate tracking fail
    {
      if (ni > track_fail_start[j] && ni < track_fail_end[j])
        imLeft = cv::Scalar(0, 0, 0);
    }
#endif

#ifdef DR
    int odom_data = 0;
    while (odom_times[dr_idx] <= (vTimeStamps[ni])) {
      cout << "current encoder time: "
           << std::to_string(odom_times[dr_idx] / 1000000.0) << endl;
      cur_pose = odom_poses[dr_idx];
      cout << "[DR] dr_idx = " << dr_idx << "pose = " << cur_pose.x << ","
           << cur_pose.y << "," << cur_pose.th << endl;
      dr_idx++;
      odom_data++;
    }
    std::cout << "current image time " << setprecision(18) << vTimeStamps[ni] << std::endl;
    if (odom_data == 0) {
      cout << "NO odom data for this image..." << endl;
      SLAM.SetOdomFlag(false);
      // continue;
    }else{ SLAM.SetOdomFlag(true);}

    if (frame_id == 0 && odom_data != 0) {
      old_pose = odom_poses[dr_idx--];
      SLAM.SetFirstDRpose(old_pose.x, old_pose.y, old_pose.th);
      frame_id++;
    }

    // update DR
    g_del_x = cur_pose.x - old_pose.x;
    g_del_y = cur_pose.y - old_pose.y;
    g_del_th = cur_pose.th - old_pose.th;

    SLAM.SetDR(cur_pose.x, cur_pose.y, cur_pose.th, g_del_x, g_del_y, g_del_th);
    old_pose = cur_pose;
#endif

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 =
        std::chrono::monotonic_clock::now();
#endif
    double tframe = vTimeStamps[ni];
    // Pass the images to the SLAM system
    SLAM.TrackStereo(imLeft, imRight, tframe);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 =
        std::chrono::monotonic_clock::now();
#endif

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)
            .count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimeStamps[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimeStamps[ni - 1];

    if (ttrack < T) usleep((T - ttrack) * 1e6);

    if (SLAM.mbStop) break;
  }

  cout << "go to stop " << endl;
  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

  // Save camera trajectory
  SLAM.SaveTrajectoryTUM(dataset_name+"CameraTrajectory.txt");
  SLAM.SaveMap(map_name);  //保存成二进制文件
  return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
  ifstream fTimes;
  fTimes.open(strPathTimes.c_str());
  vTimeStamps.reserve(5000);
  vstrImageLeft.reserve(5000);
  vstrImageRight.reserve(5000);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimeStamps.push_back(t / 1e3);
      string left_str;
      ss >> left_str;
      string right_str;
      ss >> right_str;
      vstrImageLeft.push_back(strPathLeft + "/" + left_str);
      vstrImageRight.push_back(strPathRight + "/" + right_str);
    }
  }
}

void readOdomData(string fname, vector<POSE2D>& pose_data,
                  vector<double>& pose_times) {
  ifstream fs(fname);

  string line;
  int count = 0;
  if (fs.is_open()) {
    while (getline(fs, line)) {
      std::vector<std::string> results;
      boost::split(results, line, [](char c) { return c == ' '; });
      double curtime = stod(results[0]) / 1e3;
      pose_times.push_back(curtime);

      std::string pose_str = results[1];
      std::vector<std::string> pose_result;
      boost::split(pose_result, pose_str, [](char c) { return c == ','; });
      POSE2D curpose;
      curpose.x = stod(pose_result[0]);
      curpose.y = stof(pose_result[1]);
      curpose.th = stof(pose_result[2]);

      pose_data.push_back(curpose);
      count++;
    }
    cout << "total num of odom= " << count << endl;
    fs.close();
  } else
    cout << "Unable encoder to open file" << endl;
}

void readSkipData(string fname, vector<int>& track_fail_start,
                  vector<int>& track_fail_end) {
  ifstream fs(fname);

  string line;
  int count = 0;
  if (fs.is_open()) {
    while (getline(fs, line)) {
      cout << line << '\n';

      std::vector<std::string> results;

      boost::split(results, line, [](char c) { return c == ','; });

      track_fail_start.push_back(stoi(results[0]));
      track_fail_end.push_back(stoi(results[1]));

      count++;
    }
    fs.close();
  } else
    cout << "Unable to open skip file" << endl;
}
