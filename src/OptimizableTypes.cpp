#include "OptimizableTypes.h"

namespace ORB_SLAM2 {
bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j) information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {
  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  g2o::VertexSE3Expmap* vi = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
  Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];

  Eigen::Matrix<double, 3, 6> SE3deriv;
  SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f, -z, 0.f, x, 0.f, 1.f, 0.f, y, -x, 0.f,
      0.f, 0.f, 1.f;

  _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;
}

bool EdgeSE3ProjectXYZOnlyPoseToBody::read(std::istream& is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j) information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPoseToBody::write(std::ostream& os) const {
  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus() {
  g2o::VertexSE3Expmap* vi = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
  g2o::SE3Quat T_lw(vi->estimate());
  //Xw: 世界坐标系中的坐标
  Eigen::Vector3d X_l = T_lw.map(Xw); //左目相机坐标系中的坐标
  Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));//右目相机坐标系中的坐标

  double x_w = X_l[0];
  double y_w = X_l[1];
  double z_w = X_l[2];//为什么这里用左目的世界坐标去构建反对称矩阵

  Eigen::Matrix<double, 3, 6> SE3deriv;
  SE3deriv << 0.f, z_w, -y_w, 1.f, 0.f, 0.f, -z_w, 0.f, x_w, 0.f, 1.f, 0.f, y_w,
      -x_w, 0.f, 0.f, 0.f, 1.f;

  _jacobianOplusXi =
      -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
}
}  // namespace ORB_SLAM2