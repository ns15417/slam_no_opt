#ifndef SYSTEMSETTING_H
#define SYSTEMSETTING_H

#include <string>
#include "ORBVocabulary.h"
#include<opencv2/core/core.hpp>

namespace ORB_SLAM2{
    class SystemSetting{
        public:
            SystemSetting(ORBVocabulary* pVoc);
            bool LoadSystemSetting(const std::string strSettingPath);

            public:
            ORBVocabulary* pVocabulary;

            float width;
            float height;
            float fx;
            float fy;
            float cx;
            float cy;
            float invfx;
            float invfy;
            float bf;
            float b;
            float fps;
            cv::Mat K;
            cv::Mat DistCoef;
            bool initialized;
            
            //Camera RGB parameters
            int nRGB;
            
            //ORB feature parameters
            int nFeatures;
            float fScaleFactor;
            int nLevels;
            float fIniThFAST;
            float fMinThFAST;
            
            //other parameters
            float ThDepth = -1;
            float DepthMapFactor = -1;

            //parameters for DR+SE
            bool bDR;
            float DistCamFromCenter_h;  //DIst in horizonal
            float DistCamFromCenter_v;  //Dist in vertical
            float DR_x;
            float DR_y;
            float DR_th;
            float DR_del_x;
            float DR_del_y;
            float DR_del_th;

            bool IsScaled;
            cv::Mat mTbc;
            
    };
}
#endif