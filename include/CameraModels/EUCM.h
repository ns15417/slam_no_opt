#ifndef CAMERAMODELS_EUCM_H
#define CAMERAMODELS_EUCM_H

#include <assert.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <vector>

#include "GeometricCamera.h"
using namespace std;
namespace ORB_SLAM2 {

class EUCM final : public GeometricCamera {
 public:
  EUCM() : precision(1e-6) {
    assert(mvParameters.size() == 8);
    mnType = CAM_FISHEYE_EUCM;
    toK();
  }
  EUCM(const std::vector<float> _vParameters)
      : GeometricCamera(_vParameters), precision(1e-6), mvLappingArea(2, 0) {
    assert(mvParameters.size() == 8);
    // mnId = nNextId++;
    cout << "[EUCM] check _vParameters: " << _vParameters[4] << ", "
         << _vParameters[5] << endl;
    mnType = CAM_FISHEYE_EUCM;
    toK();
  }

  EUCM(const std::vector<float> _vParameters, const float _precision)
      : GeometricCamera(_vParameters),
        precision(_precision),
        mvLappingArea(2, 0) {
    assert(mvParameters.size() == 8);
    // mnId = nNextId++;
    mnType = CAM_FISHEYE_EUCM;
    toK();
  }
  int world2Img(const cv::Point3f &p3D, cv::Point2f &uv);
  int world2Img(const cv::Mat &p3DMat, cv::Point2f &uv);
  int world2Img(const Eigen::Vector3d & v3D, Eigen::Vector2d &vImguv);

  int world2Camera(const cv::Point3f &p3D, cv::Point2f &campt);
  cv::Point3f world2Camera(const cv::Mat &p3D);
  cv::Point2f Camera2Img(cv::Point2f &p2D);
  cv::Point2f Camera2Img(cv::Point3f &p3D);
  cv::Point3f Img2Camera(cv::Point2f &uv);
  void toK();
  cv::Mat projectJac(const cv::Point3f &p3D);
  Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D);
  cv::Vec2d ComputeError(cv::Mat &P3D_Mat);
  cv::Vec2d ComputeError(cv::Point3f &p3P);
  void SetMeasurement(cv::Point2f &kp_xy);
  float chi2(const float sigma);

 public:
  float malpha;
  float mbeta;
  float mR2range;
  cv::Mat mCameraK;
  float precision;
  std::vector<int> mvLappingArea;
  cv::Vec2d mMeasurement;
  cv::Vec2d mError;
};
}  // namespace ORB_SLAM2
#endif