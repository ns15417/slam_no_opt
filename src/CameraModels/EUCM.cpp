#include "EUCM.h"
#include "cmath"

using namespace std;
namespace ORB_SLAM2 {
// world2Camera
 int EUCM::world2Camera(const cv::Point3f &p3D, cv::Point2f &campt) {
  float p3dC1x = p3D.x;
  float p3dC1y = p3D.y;
  float p3dC1z = p3D.z;
  float im1d =
      sqrt(mbeta * (p3dC1x * p3dC1x + p3dC1y * p3dC1y) + p3dC1z * p3dC1z);

  campt.x = p3dC1x / (malpha * im1d + (1 - malpha) * p3dC1z);
  campt.y = p3dC1y / (malpha * im1d + (1 - malpha) * p3dC1z);
  // EUCM model unprojection range, R2 = Mx^2+My^2
  float distance2R2 = campt.x * campt.x + campt.y * campt.y;
  if (distance2R2 > mR2range)
    return -1;
  
  return 0;
}

int EUCM::world2Img(const cv::Mat &p3DMat, cv::Point2f &uv) 
{
  cv::Point3f p3D;
  p3D.x = p3DMat.at<float>(0);  p3D.y = p3DMat.at<float>(1);
  p3D.z = p3DMat.at<float>(2);
  int ret = world2Img(p3D, uv);

  return ret;
}

int EUCM::world2Img(const cv::Point3f &p3D, cv::Point2f &uv) 
{
  float im1d = sqrt(mbeta * (p3D.x * p3D.x + p3D.y * p3D.y) + p3D.z * p3D.z);
  cv::Point2f campt;
  campt.x = p3D.x / (malpha * im1d + (1 - malpha) * p3D.z);
  campt.y = p3D.y / (malpha * im1d + (1 - malpha) * p3D.z);

  float distance2R2 = campt.x * campt.x + campt.y * campt.y;

  if (distance2R2 > mR2range)
    return -1; 
  
  cv::Mat ptincam = cv::Mat::ones(3, 1, CV_32F);
  ptincam.at<float>(0) = campt.x;
  ptincam.at<float>(1) = campt.y;
  cv::Mat ptinImg = cv::Mat::ones(3, 1, CV_32F);
  ptinImg = mCameraK * ptincam;

  uv.x = ptinImg.at<float>(0);
  uv.y = ptinImg.at<float>(1);

  return 0;
}

cv::Point3f EUCM::world2Camera(const cv::Mat &p3D) {
  float p3dC1x = p3D.at<float>(0);
  float p3dC1y = p3D.at<float>(1);
  float p3dC1z = p3D.at<float>(2);
  float im1d =
      sqrt(mbeta * (p3dC1x * p3dC1x + p3dC1y * p3dC1y) + p3dC1z * p3dC1z);

  cv::Point3f campt;
  campt.x = p3dC1x / (malpha * im1d + (1 - malpha) * p3dC1z);
  campt.y = p3dC1y / (malpha * im1d + (1 - malpha) * p3dC1z);
  float distance1R2 = campt.x * campt.x + campt.y * campt.y;
  campt.z =
      (1 - mbeta * malpha * malpha * distance1R2) /
      (malpha * sqrt(1 - (2 * malpha - 1) * mbeta * distance1R2) + 1 - malpha);
  return campt;
}

cv::Point2f EUCM::Camera2Img(cv::Point2f &p2D) {
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

cv::Point2f EUCM::Camera2Img(cv::Point3f &p3D){
  cv::Mat ptincam = cv::Mat::ones(3, 1, CV_32F);
  ptincam.at<float>(0) = p3D.x;
  ptincam.at<float>(1) = p3D.y;
  cv::Mat ptinImg = cv::Mat::ones(3, 1, CV_32F);
  ptinImg = mCameraK * ptincam;
  cv::Point2f Imgpt;
  Imgpt.x = ptinImg.at<float>(0);
  Imgpt.y = ptinImg.at<float>(1);
  return Imgpt;
}
/**
 * @brief 像素平面
 *
 * @param uv 
 * @return cv::Point3f 
 */
cv::Point3f EUCM::Img2Camera(cv::Point2f &uv){
    cv::Point3f campt;
    float fx = mvParameters[0];
    float fy = mvParameters[1];
    float cx = mvParameters[2];
    float cy = mvParameters[3];
    float alpha = mvParameters[4];
    float beta = mvParameters[5];

    float invfx = 1./fx;
    float invfy = 1./fy;
    campt.x = (uv.x - cx) * invfx;
    campt.y = (uv.y - cy) * invfy;
    float distanceR2 = campt.x * campt.x + campt.y * campt.y;
    campt.z =
        (1 - beta * alpha * alpha * distanceR2) /
        (alpha * sqrt(1 - (2 * alpha - 1) * beta * distanceR2) + 1 - alpha);
    if (isnan(campt.z))
      campt.z = (1 - beta * alpha * alpha * distanceR2) / (1 - alpha);
    return campt;
}

void EUCM::toK() {
  malpha = mvParameters[4];
  mbeta = mvParameters[5];
  mCameraK = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2],
               0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);

  mR2range = 1.0f / (mbeta * (2 * malpha - 1));
}

}  // namespace ORB_SLAM2