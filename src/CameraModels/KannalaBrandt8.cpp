#include "KannalaBrandt8.h"
#include "cmath"

using namespace std;
namespace ORB_SLAM2 {

int KannalaBrandt8::world2Img(const cv::Mat &p3DMat, cv::Point2f &uv)
{
  cv::Point3f p3D;
  p3D.x = p3DMat.at<float>(0);  p3D.y = p3DMat.at<float>(1);
  p3D.z = p3DMat.at<float>(2);
  int ret = world2Img(p3D, uv);

  return ret;
}

int KannalaBrandt8::world2Img(const cv::Point3f &p3D, cv::Point2f &uv){
  const double x2_plus_y2 = p3D.x * p3D.x + p3D.y *p3D.y;
  const double theta = atan2f(sqrtf(x2_plus_y2), p3D.z);
  const double psi = atan2f(p3D.y, p3D.x);

  const double theta2 = theta * theta;
  const double theta3 = theta * theta2;
  const double theta5 = theta3 * theta2;
  const double theta7 = theta5 * theta2;
  const double theta9 = theta7 * theta2;
  const double r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5 +
                   mvParameters[6] * theta7 + mvParameters[7] * theta9;

  Eigen::Vector2d res;
  res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
  res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];

  uv.x = res[0];
  uv.y = res[1];

  return 0;
}

int KannalaBrandt8::world2Img(const Eigen::Vector3d &v3D,
                              Eigen::Vector2d &vImguv) {
  const double x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
  const double theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
  const double psi = atan2f(v3D[1], v3D[0]);

  const double theta2 = theta * theta;
  const double theta3 = theta * theta2;
  const double theta5 = theta3 * theta2;
  const double theta7 = theta5 * theta2;
  const double theta9 = theta7 * theta2;
  const double r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5 +
                   mvParameters[6] * theta7 + mvParameters[7] * theta9;

  vImguv[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
  vImguv[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];

  return 0;
}
// world2Camera
int KannalaBrandt8::world2Camera(const cv::Point3f &p3D, cv::Point2f &campt) {
 return 0;
}

cv::Point3f KannalaBrandt8::world2Camera(const cv::Mat &p3D) {
  
  cv::Point3f campt;
  return campt;
}

cv::Point2f KannalaBrandt8::Camera2Img(cv::Point2f &p2D) {
  cv::Mat ptincam = cv::Mat::ones(3, 1, CV_32F);
  ptincam.at<float>(0) = p2D.x;
  ptincam.at<float>(1) = p2D.y;
  cv::Mat ptinImg = cv::Mat::ones(3, 1, CV_32F);
  ptinImg = mCameraK * ptincam;
  cv::Point2f Imgpt;
  Imgpt.x = ptinImg.at<float>(0);
  Imgpt.y = ptinImg.at<float>(1);
  return Imgpt;
}

cv::Point2f KannalaBrandt8::Camera2Img(cv::Point3f &p3D) {

}

/**
 * @brief 反投影:将点由像素坐标转换到归一化相机坐标中
 * 
 * @param uv 特征点的像素坐标
 * @return cv::Point3f 
 */
cv::Point3f KannalaBrandt8::Img2Camera(cv::Point2f &uv) {
  cv::Point2f pw((uv.x - mvParameters[2]) / mvParameters[0],
                 (uv.y - mvParameters[3]) / mvParameters[1]);
  float scale = 1.f;
  float theta_d = sqrtf(pw.x * pw.x + pw.y * pw.y);
  theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

  if (theta_d > 1e-8) {
    // Compensate distortion iteratively
    float theta = theta_d;

    for (int j = 0; j < 10; j++) {
      float theta2 = theta * theta, theta4 = theta2 * theta2,
            theta6 = theta4 * theta2, theta8 = theta4 * theta4;
      float k0_theta2 = mvParameters[4] * theta2,
            k1_theta4 = mvParameters[5] * theta4;
      float k2_theta6 = mvParameters[6] * theta6,
            k3_theta8 = mvParameters[7] * theta8;
      float theta_fix =
          (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) -
           theta_d) /
          (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
      theta = theta - theta_fix;
      if (fabsf(theta_fix) < precision) break;
    }
    // scale = theta - theta_d;
    scale = std::tan(theta) / theta_d;
  }

  return cv::Point3f(pw.x * scale, pw.y * scale, 1.f);
}

void KannalaBrandt8::toK() {
  mCameraK = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2],
               0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
}

cv::Mat KannalaBrandt8::projectJac(const cv::Point3f &p3D) {
  float x2 = p3D.x * p3D.x, y2 = p3D.y * p3D.y, z2 = p3D.z * p3D.z;
  float r2 = x2 + y2;
  float r = sqrt(r2);
  float r3 = r2 * r;
  float theta = atan2(r, p3D.z);

  float theta2 = theta * theta, theta3 = theta2 * theta;
  float theta4 = theta2 * theta2, theta5 = theta4 * theta;
  float theta6 = theta2 * theta4, theta7 = theta6 * theta;
  float theta8 = theta4 * theta4, theta9 = theta8 * theta;

  float f = theta + theta3 * mvParameters[4] + theta5 * mvParameters[5] +
            theta7 * mvParameters[6] + theta9 * mvParameters[7];
  float fd = 1 + 3 * mvParameters[4] * theta2 + 5 * mvParameters[5] * theta4 +
             7 * mvParameters[6] * theta6 + 9 * mvParameters[7] * theta8;

  cv::Mat Jac(2, 3, CV_32F);
  Jac.at<float>(0, 0) =
      mvParameters[0] * (fd * p3D.z * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
  Jac.at<float>(1, 0) =
      mvParameters[1] *
      (fd * p3D.z * p3D.y * p3D.x / (r2 * (r2 + z2)) - f * p3D.y * p3D.x / r3);

  Jac.at<float>(0, 1) =
      mvParameters[0] *
      (fd * p3D.z * p3D.y * p3D.x / (r2 * (r2 + z2)) - f * p3D.y * p3D.x / r3);
  Jac.at<float>(1, 1) =
      mvParameters[1] * (fd * p3D.z * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

  Jac.at<float>(0, 2) = -mvParameters[0] * fd * p3D.x / (r2 + z2);
  Jac.at<float>(1, 2) = -mvParameters[1] * fd * p3D.y / (r2 + z2);

  std::cout << "CV JAC: " << Jac << std::endl;

  return Jac.clone();
}

Eigen::Matrix<double, 2, 3> KannalaBrandt8::projectJac(const Eigen::Vector3d &v3D) {
  double x2 = v3D[0] * v3D[0], y2 = v3D[1] * v3D[1], z2 = v3D[2] * v3D[2];
  double r2 = x2 + y2;
  double r = sqrt(r2);
  double r3 = r2 * r;
  double theta = atan2(r, v3D[2]);

  double theta2 = theta * theta, theta3 = theta2 * theta;
  double theta4 = theta2 * theta2, theta5 = theta4 * theta;
  double theta6 = theta2 * theta4, theta7 = theta6 * theta;
  double theta8 = theta4 * theta4, theta9 = theta8 * theta;

  double f = theta + theta3 * mvParameters[4] + theta5 * mvParameters[5] +
             theta7 * mvParameters[6] + theta9 * mvParameters[7];
  double fd = 1 + 3 * mvParameters[4] * theta2 + 5 * mvParameters[5] * theta4 +
              7 * mvParameters[6] * theta6 + 9 * mvParameters[7] * theta8;

  Eigen::Matrix<double, 2, 3> JacGood;
  JacGood(0, 0) =
      mvParameters[0] * (fd * v3D[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
  JacGood(1, 0) =
      mvParameters[1] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) -
                         f * v3D[1] * v3D[0] / r3);

  JacGood(0, 1) =
      mvParameters[0] * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) -
                         f * v3D[1] * v3D[0] / r3);
  JacGood(1, 1) =
      mvParameters[1] * (fd * v3D[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

  JacGood(0, 2) = -mvParameters[0] * fd * v3D[0] / (r2 + z2);
  JacGood(1, 2) = -mvParameters[1] * fd * v3D[1] / (r2 + z2);

  return JacGood;
}

// cv::Mat KannalaBrandt8::unprojectJac(const cv::Point2f &p2D) {
//   return cv::Mat();
// }

cv::Vec2d KannalaBrandt8::ComputeError(cv::Point3f &p3P){
  cv::Point2f obs;
  world2Img(p3P,obs);
  mError[0] = obs.x - mMeasurement[0];
  mError[1] = obs.y - mMeasurement[1];

  return mError;
}

cv::Vec2d KannalaBrandt8::ComputeError(cv::Mat &P3D_Mat){
  cv::Point3f pos;
  pos.x = P3D_Mat.at<float>(0);
  pos.y = P3D_Mat.at<float>(1);
  pos.z = P3D_Mat.at<float>(2);

  cv::Vec2d error = ComputeError(pos);

  return error;
}

void KannalaBrandt8::SetMeasurement(cv::Point2f &kp_xy){
  mMeasurement[0] = kp_xy.x;
  mMeasurement[1] = kp_xy.y;
}

float KannalaBrandt8::chi2(const float sigma){
  cv::Vec2d b = sigma*mError;

  return mError.dot(b);
}

}  // namespace ORB_SLAM2