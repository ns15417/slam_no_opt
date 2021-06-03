#ifndef CAMERAMODELS_EUCM_H
#define CAMERAMODELS_EUCM_H

#include <assert.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "GeometricCamera.h"
using namespace std;
namespace ORB_SLAM2{

class EUCM final : public GeometricCamera{

public:
EUCM():precision(1e-6){
  assert(mvParameters.size() == 8);
  // mnId = nNextId++;
  mnType = CAM_FISHEYE_EUCM;
}
EUCM(const std::vector<float> _vParameters) : GeometricCamera(_vParameters), precision(1e-6), mvLappingArea(2, 0) {
  assert(mvParameters.size() == 8);
  // mnId = nNextId++;
  cout << "[EUCM] check _vParameters: " << _vParameters[4] << ", " << _vParameters[5]<< endl;
  mnType = CAM_FISHEYE_EUCM;
}

EUCM(const std::vector<float> _vParameters, const float _precision) : GeometricCamera(_vParameters),precision(_precision),mvLappingArea(2, 0) {
  assert(mvParameters.size() == 8);
  // mnId = nNextId++;
  mnType = CAM_FISHEYE_EUCM;
}

cv::Point2f world2Camera(const cv::Point3f &p3D);
cv::Point3f world2Camera(const cv::Mat &p3D);
cv::Point2f Camera2Img(cv::Point2f &p2D);
cv::Point2f Camera2Img(cv::Point3f &p3D);
cv::Point3f Img2Camera(cv::Point2f &uv);
cv::Mat toK();

public:
float malpha;
float mbeta;
cv::Mat mCameraK;
float precision;
std::vector<int> mvLappingArea;
};
}
#endif 