/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CAMERAMODELS_GEOMETRICCAMERA_H
#define CAMERAMODELS_GEOMETRICCAMERA_H

#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Geometry>

namespace ORB_SLAM2 {

    class GeometricCamera {
    public:
        GeometricCamera() {}
        GeometricCamera(const std::vector<float> &_vParameters) : mvParameters(_vParameters) {}
        ~GeometricCamera() {}

        /**
         * @brief projection function
         * 
         * @param p3D :object location in camera coordination
         * @param uv :object location in pixel coordination after projection
         * @return int 
         * @note Coordinate is needed to be transformed to camera coordination
         *       when using points in world coordination
         */
        virtual int world2Img(const cv::Point3f &p3D, cv::Point2f &uv) = 0;
        virtual int world2Img(const cv::Mat &p3DMat, cv::Point2f &uv) = 0;
        virtual int world2Img(const Eigen::Vector3d & v3D, Eigen::Vector2d &vImguv) = 0;

        virtual int world2Camera(const cv::Point3f &p3D, cv::Point2f &campt) = 0;
        virtual cv::Point3f world2Camera(const cv::Mat &p3D) = 0;
        virtual cv::Point2f Camera2Img(cv::Point2f &p2D) = 0;
        virtual cv::Point2f Camera2Img(cv::Point3f &p3D) = 0;
        virtual cv::Point3f Img2Camera(cv::Point2f &uv) = 0;
        virtual void toK() = 0;
        
        virtual cv::Mat projectJac(const cv::Point3f &p3D) = 0;
        virtual Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D) = 0;

        virtual void SetMeasurement(cv::Point2f &kp_xy) = 0;
        virtual float chi2(const float sigma) = 0;
        
        virtual cv::Vec2d ComputeError(cv::Point3f &p3P) = 0;
        virtual cv::Vec2d ComputeError(cv::Mat &P3D_Mat) = 0;

        // virtual bool epipolarConstrain(GeometricCamera* otherCamera, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc) = 0;

        float getParameter(const int i){return mvParameters[i];}
        void setParameter(const float p, const size_t i){mvParameters[i] = p;}

        unsigned int GetId() { return mnId; }

        unsigned int GetType() { return mnType; }

        const unsigned int CAM_PINHOLE = 0;
        const unsigned int CAM_FISHEYE_KB = 1;
        const unsigned int CAM_FISHEYE_EUCM = 2;

        // static long unsigned int nNextId;

    public:
        
        std::vector<float> mvParameters;

        unsigned int mnId;

        unsigned int mnType;
    };
}


#endif //CAMERAMODELS_GEOMETRICCAMERA_H
