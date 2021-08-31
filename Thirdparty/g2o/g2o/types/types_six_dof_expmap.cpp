// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "types_six_dof_expmap.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

using namespace std;


Vector2d project2d(const Vector3d& v)  {
  Vector2d res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector3d unproject2d(const Vector2d& v)  {
  Vector3d res;
  res(0) = v(0);
  res(1) = v(1);
  res(2) = 1;
  return res;
}

VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
}

bool VertexSE3Expmap::read(std::istream& is) {
  Vector7d est;
  for (int i=0; i<7; i++)
    is  >> est[i];
  SE3Quat cam2world;
  cam2world.fromVector(est);
  setEstimate(cam2world.inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream& os) const {
  SE3Quat cam2world(estimate().inverse());
  for (int i=0; i<7; i++)
    os << cam2world[i] << " ";
  return os.good();
}


EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


/*void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z; 
  
  Matrix<double,2,3> tmp;
  tmp(0,0) = fx;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*fx;

  tmp(1,0) = 0;
  tmp(1,1) = fy;
  tmp(1,2) = -y/z*fy;

  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;
}*/

void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double x_2 = x*x;
  double y_2 = y*y;
  double z_2 = z*z;

  double rho = sqrt( beta*(x_2+y_2)+z_2 );
  double eta = (1-alpha)*z + alpha*rho;
  double eta_2 = eta*eta;
  
  Matrix<double,2,3> tmp;
  tmp(0,0) = fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) );
  tmp(0,1) = fx * (alpha*beta*x*y) / (eta_2*rho);
  tmp(0,2) = fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;

  tmp(1,0) = fy * (alpha*beta*x*y) / (eta_2*rho);
  tmp(1,1) = fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) );
  tmp(1,2) = fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;

  _jacobianOplusXi =  tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) = -z * fx * (alpha*beta*x*y) / (eta_2*rho) + y * fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXj(0,1) = z * fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) ) - x * fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXj(0,2) = -y * fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) ) + x * fx * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXj(0,3) = fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) );
  _jacobianOplusXj(0,4) = fx * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXj(0,5) = fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;

  _jacobianOplusXj(1,0) = -z * fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) ) + y * fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXj(1,1) = z * fy * (alpha*beta*x*y) / (eta_2*rho) - x * fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXj(1,2) = -y * fy * (alpha*beta*x*y) / (eta_2*rho) + x * fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) );
  _jacobianOplusXj(1,3) = fy * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXj(1,4) = fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) );
  _jacobianOplusXj(1,5) = fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
}

/*Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}*/

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj;
  Vector2d res;
  const float imd = sqrt( beta*(trans_xyz[0]*trans_xyz[0]+trans_xyz[1]*trans_xyz[1])+trans_xyz[2]*trans_xyz[2] );
  proj[0] = trans_xyz[0] / ( alpha*imd+(1-alpha)*trans_xyz[2] );
  proj[1] = trans_xyz[1] / ( alpha*imd+(1-alpha)*trans_xyz[2] );
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


Vector4d EdgeStereoSE3ProjectXYZ::cam_project(const Vector3d& trans_xyz) const {
  Vector4d res;
  Vector2d proj;
  const float imd =
      sqrt(beta * (trans_xyz[0] * trans_xyz[0] + trans_xyz[1] * trans_xyz[1]) +
           trans_xyz[2] * trans_xyz[2]);
  proj[0] = trans_xyz[0] / (alpha * imd + (1 - alpha) * trans_xyz[2]);
  proj[1] = trans_xyz[1] / (alpha * imd + (1 - alpha) * trans_xyz[2]);

  res[0] = proj[0] * fx + cx;  
  res[1] = proj[1] * fy + cy; 

  Vector3d right_xyz = R21 * trans_xyz + t2;
  Vector2d proj_r;
  const float r_imd = sqrt(
      r_beta * (right_xyz[0] * right_xyz[0] + right_xyz[1] * right_xyz[1]) +
      right_xyz[2] * right_xyz[2]);
  proj_r[0] = right_xyz[0] / (r_alpha * r_imd + (1 - r_alpha) * right_xyz[2]);
  proj_r[1] = right_xyz[1] / (r_alpha * r_imd + (1 - r_alpha) * right_xyz[2]);
  res[2] = proj_r[0] * r_fx + r_cx;
  res[3] = proj_r[1] * r_fy + r_cy;

  return res;
}

EdgeStereoSE3ProjectXYZ::EdgeStereoSE3ProjectXYZ()
    : BaseBinaryEdge<4, Vector4d, VertexSBAPointXYZ, VertexSE3Expmap>() {}

bool EdgeStereoSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap* vj = static_cast<VertexSE3Expmap*>(_vertices[1]);
  SE3Quat T(vj->estimate()); 
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate(); 
  Vector3d xyz_trans =
      T.map(xyz);  

  const Matrix3d R = T.rotation().toRotationMatrix();
  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double x_2 = x * x;
  double y_2 = y * y;
  double z_2 = z * z;

  double rho = sqrt(beta * (x_2 + y_2) + z_2);
  double eta = (1 - alpha) * z + alpha * rho;
  double eta_2 = eta * eta;

  _jacobianOplusXj(0, 0) = -z * fx * (alpha * beta * x * y) / (eta_2 * rho) +
                           y * fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXj(0, 1) =
      z * fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho)) -
      x * fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXj(0, 2) =
      -y * fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho)) +
      x * fx * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXj(0, 3) =
      fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho));
  _jacobianOplusXj(0, 4) = fx * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXj(0, 5) = fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;

  _jacobianOplusXj(1, 0) =
      -z * fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho)) +
      y * fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXj(1, 1) = z * fy * (alpha * beta * x * y) / (eta_2 * rho) -
                           x * fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXj(1, 2) =
      -y * fy * (alpha * beta * x * y) / (eta_2 * rho) +
      x * fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho));
  _jacobianOplusXj(1, 3) = fy * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXj(1, 4) =
      fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho));
  _jacobianOplusXj(1, 5) = fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;

  Vector3d r_xyz_trans = R21 * xyz_trans + t2;
  double r_x = r_xyz_trans[0];
  double r_y = r_xyz_trans[1];
  double r_z = r_xyz_trans[2];
  double r_x2 = r_x * r_x;
  double r_y2 = r_y * r_y;
  double r_z2 = r_z * r_z;

  double r_rho = sqrt(r_beta * (r_x2 + r_y2) + r_z2);
  double r_eta = (1 - r_alpha) * r_z + r_alpha * r_rho;
  double r_eta2 = r_eta * r_eta;

  double r_rho_x =
      (r_beta * (R21(0, 0) * r_x + R21(1, 0) * r_y) + r_z * R21(2, 0)) / r_rho;
  double r_rho_y =
      (r_beta * (R21(0, 1) * r_x + R21(1, 1) * r_y) + r_z * R21(2, 1)) / r_rho;
  double r_rho_z =
      (r_beta * (R21(0, 2) * r_x + R21(1, 2) * r_y) + r_z * R21(2, 2)) / r_rho;

  double r_eta_x = r_alpha * r_rho_x + (1 - r_alpha) * R21(2, 0);
  double r_eta_y = r_alpha * r_rho_y + (1 - r_alpha) * R21(2, 1);
  double r_eta_z = r_alpha * r_rho_z + (1 - r_alpha) * R21(2, 2);

  double r_u_x = (r_fx * R21(0, 0) * r_eta - r_fx * r_x * r_eta_x) / (r_eta2);
  double r_u_y = (r_fx * R21(0, 1) * r_eta - r_fx * r_x * r_eta_y) / (r_eta2);
  double r_u_z = (r_fx * R21(0, 2) * r_eta - r_fx * r_x * r_eta_z) / (r_eta2);

  double r_v_x = (r_fy * R21(1, 0) * r_eta - r_fy * r_y * r_eta_x) / (r_eta2);
  double r_v_y = (r_fy * R21(1, 1) * r_eta - r_fy * r_y * r_eta_y) / (r_eta2);
  double r_v_z = (r_fy * R21(1, 2) * r_eta - r_fy * r_y * r_eta_z) / (r_eta2);

  _jacobianOplusXj(2, 0) = z * r_u_y - y * r_u_z;
  _jacobianOplusXj(2, 1) = x * r_u_z - z * r_u_x;
  _jacobianOplusXj(2, 2) = y * r_u_x - x * r_u_y;
  _jacobianOplusXj(2, 3) = -r_u_x;
  _jacobianOplusXj(2, 4) = -r_u_y;
  _jacobianOplusXj(2, 5) = -r_u_z;

  _jacobianOplusXj(3, 0) = z * r_v_y - y * r_v_z;
  _jacobianOplusXj(3, 1) = x * r_v_z - z * r_v_x;
  _jacobianOplusXj(3, 2) = r_v_x * y - r_v_y * x;
  _jacobianOplusXj(3, 3) = -r_v_x;
  _jacobianOplusXj(3, 4) = -r_v_y;
  _jacobianOplusXj(3, 5) = -r_v_z;

  _jacobianOplusXi(0, 0) = _jacobianOplusXj(0, 3) * R(0, 0) +
                           _jacobianOplusXj(0, 4) * R(1, 0) +
                           _jacobianOplusXj(0, 5) * R(2, 0);
  _jacobianOplusXi(0, 1) = _jacobianOplusXj(0, 3) * R(0, 1) +
                           _jacobianOplusXj(0, 4) * R(1, 1) +
                           _jacobianOplusXj(0, 5) * R(2, 1);
  _jacobianOplusXi(0, 2) = _jacobianOplusXj(0, 3) * R(0, 2) +
                           _jacobianOplusXj(0, 4) * R(1, 2) +
                           _jacobianOplusXj(0, 5) * R(2, 2);

  _jacobianOplusXi(1, 0) = _jacobianOplusXj(1, 3) * R(0, 0) +
                           _jacobianOplusXj(1, 4) * R(1, 0) +
                           _jacobianOplusXj(1, 5) * R(2, 0);
  _jacobianOplusXi(1, 1) = _jacobianOplusXj(1, 3) * R(0, 1) +
                           _jacobianOplusXj(1, 4) * R(1, 1) +
                           _jacobianOplusXj(1, 5) * R(2, 1);
  _jacobianOplusXi(1, 2) = _jacobianOplusXj(1, 3) * R(0, 2) +
                           _jacobianOplusXj(1, 4) * R(1, 2) +
                           _jacobianOplusXj(1, 5) * R(2, 2);

  _jacobianOplusXi(2, 0) = _jacobianOplusXj(2, 3) * R(0, 0) +
                           _jacobianOplusXj(2, 4) * R(1, 0) +
                           _jacobianOplusXj(2, 5) * R(2, 0);
  _jacobianOplusXi(2, 1) = _jacobianOplusXj(2, 3) * R(0, 1) +
                           _jacobianOplusXj(2, 4) * R(1, 1) +
                           _jacobianOplusXj(2, 5) * R(2, 1);
  _jacobianOplusXi(2, 2) = _jacobianOplusXj(2, 3) * R(0, 2) +
                           _jacobianOplusXj(2, 4) * R(1, 2) +
                           _jacobianOplusXj(2, 5) * R(2, 2);

  _jacobianOplusXi(3, 0) = _jacobianOplusXj(3, 3) * R(0, 0) +
                           _jacobianOplusXj(3, 4) * R(1, 0) +
                           _jacobianOplusXj(3, 5) * R(2, 0);
  _jacobianOplusXi(3, 1) = _jacobianOplusXj(3, 3) * R(0, 1) +
                           _jacobianOplusXj(3, 4) * R(1, 1) +
                           _jacobianOplusXj(3, 5) * R(2, 1);
  _jacobianOplusXi(3, 2) = _jacobianOplusXj(3, 3) * R(0, 2) +
                           _jacobianOplusXj(3, 4) * R(1, 2) +
                           _jacobianOplusXj(3, 5) * R(2, 2);
}

//Only Pose
bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double x_2 = x*x;
  double y_2 = y*y;
  double z_2 = z*z;
  
  double rho = sqrt( beta*(x_2+y_2)+z_2 );
  double eta = (1-alpha)*z + alpha*rho;
  double eta_2 = eta*eta;

  _jacobianOplusXi(0,0) = -z * fx * (alpha*beta*x*y) / (eta_2*rho) + y * fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXi(0,1) = z * fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) ) - x * fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXi(0,2) = -y * fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) ) + x * fx * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXi(0,3) = fx * ( -1/eta + (alpha*beta*x_2) / (eta_2*rho) );
  _jacobianOplusXi(0,4) = fx * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXi(0,5) = fx * x * ( 1-alpha+(alpha*z)/rho ) / eta_2;

  _jacobianOplusXi(1,0) = -z * fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) ) + y * fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXi(1,1) = z * fy * (alpha*beta*x*y) / (eta_2*rho) - x * fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
  _jacobianOplusXi(1,2) = -y * fy * (alpha*beta*x*y) / (eta_2*rho) + x * fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) );
  _jacobianOplusXi(1,3) = fy * (alpha*beta*x*y) / (eta_2*rho);
  _jacobianOplusXi(1,4) = fy * ( -1/eta + (alpha*beta*y_2) / (eta_2*rho) );
  _jacobianOplusXi(1,5) = fy * y * ( 1-alpha+(alpha*z)/rho ) / eta_2;
}

/*Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}*/

Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj;
  Vector2d res;
  const float imd = sqrt( beta*(trans_xyz[0]*trans_xyz[0]+trans_xyz[1]*trans_xyz[1])+trans_xyz[2]*trans_xyz[2] );
  proj[0] = trans_xyz[0] / ( alpha*imd+(1-alpha)*trans_xyz[2] );
  proj[1] = trans_xyz[1] / ( alpha*imd+(1-alpha)*trans_xyz[2] );
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

Vector4d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector4d res;
  Vector2d proj;
  const float imd =
      sqrt(beta * (trans_xyz[0] * trans_xyz[0] + trans_xyz[1] * trans_xyz[1]) +
           trans_xyz[2] * trans_xyz[2]);
  proj[0] = trans_xyz[0] / (alpha * imd + (1 - alpha) * trans_xyz[2]);
  proj[1] = trans_xyz[1] / (alpha * imd + (1 - alpha) * trans_xyz[2]);

  res[0] = proj[0] * fx + cx;  
  res[1] = proj[1] * fy + cy; 

  Vector3d right_xyz = R21 * trans_xyz + t2;
  Vector2d proj_r;
  const float r_imd = sqrt(
      r_beta * (right_xyz[0] * right_xyz[0] + right_xyz[1] * right_xyz[1]) +
      right_xyz[2] * right_xyz[2]);
  proj_r[0] = right_xyz[0] / (r_alpha * r_imd + (1 - r_alpha) * right_xyz[2]);
  proj_r[1] = right_xyz[1] / (r_alpha * r_imd + (1 - r_alpha) * right_xyz[2]);
  res[2] = proj_r[0] * r_fx + r_cx;
  res[3] = proj_r[1] * r_fy + r_cy;

  return res;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw); 
  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double x_2 = x * x;
  double y_2 = y * y;
  double z_2 = z * z;

  double rho = sqrt(beta * (x_2 + y_2) + z_2);
  double eta = (1 - alpha) * z + alpha * rho;
  double eta_2 = eta * eta;

  _jacobianOplusXi(0, 0) = -z * fx * (alpha * beta * x * y) / (eta_2 * rho) +
                           y * fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXi(0, 1) =
      z * fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho)) -
      x * fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXi(0, 2) =
      -y * fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho)) +
      x * fx * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXi(0, 3) =
      fx * (-1 / eta + (alpha * beta * x_2) / (eta_2 * rho));
  _jacobianOplusXi(0, 4) = fx * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXi(0, 5) = fx * x * (1 - alpha + (alpha * z) / rho) / eta_2;

  _jacobianOplusXi(1, 0) =
      -z * fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho)) +
      y * fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXi(1, 1) = z * fy * (alpha * beta * x * y) / (eta_2 * rho) -
                           x * fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;
  _jacobianOplusXi(1, 2) =
      -y * fy * (alpha * beta * x * y) / (eta_2 * rho) +
      x * fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho));
  _jacobianOplusXi(1, 3) = fy * (alpha * beta * x * y) / (eta_2 * rho);
  _jacobianOplusXi(1, 4) =
      fy * (-1 / eta + (alpha * beta * y_2) / (eta_2 * rho));
  _jacobianOplusXi(1, 5) = fy * y * (1 - alpha + (alpha * z) / rho) / eta_2;

  Vector3d r_xyz_trans = R21 * xyz_trans + t2;
  double r_x = r_xyz_trans[0];
  double r_y = r_xyz_trans[1];
  double r_z = r_xyz_trans[2];
  double r_x2 = r_x * r_x;
  double r_y2 = r_y * r_y;
  double r_z2 = r_z * r_z;

  double r_rho = sqrt(r_beta * (r_x2 + r_y2) + r_z2);
  double r_eta = (1 - r_alpha) * r_z + r_alpha * r_rho;
  double r_eta2 = r_eta * r_eta;

  double r_rho_x =
      (r_beta * (R21(0, 0) * r_x + R21(1, 0) * r_y) + r_z * R21(2, 0)) / r_rho;
  double r_rho_y =
      (r_beta * (R21(0, 1) * r_x + R21(1, 1) * r_y) + r_z * R21(2, 1)) / r_rho;
  double r_rho_z =
      (r_beta * (R21(0, 2) * r_x + R21(1, 2) * r_y) + r_z * R21(2, 2)) / r_rho;

  double r_eta_x = r_alpha * r_rho_x + (1 - r_alpha) * R21(2, 0);
  double r_eta_y = r_alpha * r_rho_y + (1 - r_alpha) * R21(2, 1);
  double r_eta_z = r_alpha * r_rho_z + (1 - r_alpha) * R21(2, 2);

  double r_u_x = (r_fx * R21(0, 0) * r_eta - r_fx * r_x * r_eta_x) / (r_eta2);
  double r_u_y = (r_fx * R21(0, 1) * r_eta - r_fx * r_x * r_eta_y) / (r_eta2);
  double r_u_z = (r_fx * R21(0, 2) * r_eta - r_fx * r_x * r_eta_z) / (r_eta2);

  double r_v_x = (r_fy * R21(1, 0) * r_eta - r_fy * r_y * r_eta_x) / (r_eta2);
  double r_v_y = (r_fy * R21(1, 1) * r_eta - r_fy * r_y * r_eta_y) / (r_eta2);
  double r_v_z = (r_fy * R21(1, 2) * r_eta - r_fy * r_y * r_eta_z) / (r_eta2);

  _jacobianOplusXi(2, 0) = z * r_u_y - y * r_u_z;
  _jacobianOplusXi(2, 1) = x * r_u_z - z * r_u_x;
  _jacobianOplusXi(2, 2) = y * r_u_x - x * r_u_y;
  _jacobianOplusXi(2, 3) = -r_u_x;
  _jacobianOplusXi(2, 4) = -r_u_y;
  _jacobianOplusXi(2, 5) = -r_u_z;

  _jacobianOplusXi(3, 0) = z * r_v_y - y * r_v_z;
  _jacobianOplusXi(3, 1) = x * r_v_z - z * r_v_x;
  _jacobianOplusXi(3, 2) = r_v_x * y - r_v_y * x;
  _jacobianOplusXi(3, 3) = -r_v_x;
  _jacobianOplusXi(3, 4) = -r_v_y;
  _jacobianOplusXi(3, 5) = -r_v_z;
}


} // end namespace
