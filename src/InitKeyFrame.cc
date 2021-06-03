#include "InitKeyFrame.h"
#include <opencv2/opencv.hpp>
#include "SystemSetting.h"

namespace ORB_SLAM2
{
//将参数加载到该类中
InitKeyFrame::InitKeyFrame(SystemSetting &SS):
    pVocabulary(SS.pVocabulary)//, pKeyFrameDatabase(SS.pKeyFrameDatabase)
{
    fx = SS.fx;
    fy = SS.fy;
    cx = SS.cx;
    cy = SS.cy;
    invfx = SS.invfx;
    invfy = SS.invfy;
    bf = SS.bf;
    b  = SS.b;
    ThDepth = SS.ThDepth;

    nScaleLevels = SS.nLevels;
    fScaleFactor = SS.fScaleFactor;
    fLogScaleFactor = log(SS.fScaleFactor);
    vScaleFactors.resize(nScaleLevels);
    vLevelSigma2.resize(nScaleLevels);
    vScaleFactors[0] = 1.0f;
    vLevelSigma2[0]  = 1.0f;
    for ( int i = 1; i < nScaleLevels; i ++ )
    {
        vScaleFactors[i] = vScaleFactors[i-1]*fScaleFactor;
        vLevelSigma2[i]  = vScaleFactors[i]*vScaleFactors[i];
    }
    
    vInvScaleFactors.resize(nScaleLevels);
    vInvLevelSigma2.resize(nScaleLevels);
    for ( int i = 0; i < nScaleLevels; i ++ )
    {
        vInvScaleFactors[i] = 1.0f/vScaleFactors[i];
        vInvLevelSigma2[i]  = 1.0f/vLevelSigma2[i];
    }

    K = SS.K;

    DistCoef = SS.DistCoef;
    alpha = DistCoef.at<float>(0);
    beta = DistCoef.at<float>(1);

    //TODO: not sure about whether do i need this    
    if( SS.DistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0) = 0.0;
        mat.at<float>(0,1) = 0.0;
        mat.at<float>(1,0) = SS.width;
        mat.at<float>(1,1) = 0.0;
        mat.at<float>(2,0) = 0.0;
        mat.at<float>(2,1) = SS.height;
        mat.at<float>(3,0) = SS.width;
        mat.at<float>(3,1) = SS.height;
        
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, SS.K, SS.DistCoef, cv::Mat(), SS.K);
        mat = mat.reshape(1);

        nMinX = min(mat.at<float>(0,0), mat.at<float>(2,0));
        nMaxX = max(mat.at<float>(1,0), mat.at<float>(3,0));
        nMinY = min(mat.at<float>(0,1), mat.at<float>(1,1));
        nMaxY = max(mat.at<float>(2,1), mat.at<float>(3,1));
    }
    else
    {
        nMinX = 0.0f;
        nMaxX = SS.width;
        nMinY = 0.0f;
        nMaxY = SS.height;
    }

    fGridElementWidthInv=static_cast<float>(KEYFRAME_GRID_COLS)/(nMaxX-nMinX);
    fGridElementHeightInv=static_cast<float>(KEYFRAME_GRID_ROWS)/(nMaxY-nMinY);
    
    //重定位之后先暂时不考虑DR
    //fDRX = dr_x;
    //fDRY = dr_y;
}
//for EUCM MODEL, to get vKpsUn and vP3M
void InitKeyFrame::UndistortKeyPoints()
{
    if(DistCoef.at<float>(0) == 0 )
    {
        vKpsUn = vKps;
        return;
    }

    vKpsUn.resize(N);
    vP3M.resize(N);
    for(int i=0; i<N;i++)
    {
        vP3M[i].x = (vKps[i].pt.x-cx) * invfx;
        vP3M[i].y = (vKps[i].pt.y-cy) * invfy;
        float distanceR2 = vP3M[i].x*vP3M[i].x + vP3M[i].y*vP3M[i].y;
        vP3M[i].z = (1-beta*alpha*alpha*distanceR2) / (alpha*sqrt(1-(2*alpha-1)*beta*distanceR2) + 1-alpha);
        if( isnan(vP3M[i].z) )
            vP3M[i].z = (1-beta*alpha*alpha*distanceR2) / (1-alpha);
        
        cv::KeyPoint kp = vKps[i];
        kp.pt.x=vKps[i].pt.x;
        kp.pt.y=vKps[i].pt.y;
        vKpsUn[i]=kp;
    }
}

void InitKeyFrame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(KEYFRAME_GRID_COLS*KEYFRAME_GRID_ROWS);
    for ( unsigned int i = 0; i < KEYFRAME_GRID_COLS; i ++ )
    {
        for ( unsigned int j = 0; j < KEYFRAME_GRID_ROWS; j ++)
            vGrid[i][j].reserve(nReserve);
    }
    
    for ( int i = 0; i < N; i ++ )
    {
        const cv::KeyPoint& kp = vKpsUn[i];
        int nGridPosX, nGridPosY;
        if( PosInGrid(kp, nGridPosX, nGridPosY))
            vGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

bool InitKeyFrame::PosInGrid(const cv::KeyPoint &kp, int &posX,  int &posY)
{
    posX = round((kp.pt.x-nMinX)*fGridElementWidthInv);
    posY = round((kp.pt.y-nMinY)*fGridElementHeightInv);

    if(posX<0 || posX>=KEYFRAME_GRID_COLS ||posY<0 || posY>=KEYFRAME_GRID_ROWS)
        return false;
    return true;
}

}