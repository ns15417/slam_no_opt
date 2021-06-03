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

#include "Map.h"

#include <mutex>

namespace ORB_SLAM2 {

Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0), mnKFAfterReinit(3) {}

void Map::AddKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexMap);
  if(mspKeyFrames.empty()){
    cout << "First KF:" << pKF->mnId << "; Map init KF:" << mnInitKFid << endl;
    mnInitKFid = pKF->mnId;
  }
  mspKeyFrames.insert(pKF);
  if (pKF->mnId > mnMaxKFid) mnMaxKFid = pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP) {
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP) {
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.erase(pKF);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
  unique_lock<mutex> lock(mMutexMap);
  mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange() {
  unique_lock<mutex> lock(mMutexMap);
  mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx() {
  unique_lock<mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

vector<KeyFrame *> Map::GetAllKeyFrames() {
  unique_lock<mutex> lock(mMutexMap);
  return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<MapPoint *> Map::GetAllMapPoints() {
  unique_lock<mutex> lock(mMutexMap);
  return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap() {
  unique_lock<mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap() {
  unique_lock<mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

vector<MapPoint *> Map::GetReferenceMapPoints() {
  unique_lock<mutex> lock(mMutexMap);
  return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid() {
  unique_lock<mutex> lock(mMutexMap);
  return mnMaxKFid;
}

long unsigned int Map::GetInitKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnInitKFid;
}

void Map::clear() {
  for (set<MapPoint *>::iterator sit = mspMapPoints.begin(),
                                 send = mspMapPoints.end();
       sit != send; sit++)
    delete *sit;

  for (set<KeyFrame *>::iterator sit = mspKeyFrames.begin(),
                                 send = mspKeyFrames.end();
       sit != send; sit++)
    delete *sit;

  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpReferenceMapPoints.clear();
  mvpKeyFrameOrigins.clear();
  mnKFAfterReinit = 3;
}

long unsigned int Map::KeyFramesAfterReinit() {
  unique_lock<mutex> lock(mMutexMap);

  return mnKFAfterReinit;
}

void Map::ResetKFCounter() {
  unique_lock<mutex> lock(mMutexMap);
  mnKFAfterReinit = 0;
}

void Map::IncreaseKFCounter() {
  unique_lock<mutex> lock(mMutexMap);
  mnKFAfterReinit++;
}

// SAVE THE WHOLE MAP FOR OFFLINE RELOCALIZATION
void Map::Save(const std::string filename)
{
    ofstream f;
    f.open(filename.c_str(), ios_base::out | ios::binary);
    cerr << "The number of Mappoints is " << mspMapPoints.size() << endl;

    //写入地图点个数
    unsigned long int nMapPoints = mspMapPoints.size();
    f.write((char*)&nMapPoints, sizeof(nMapPoints));
    for(auto mp:mspMapPoints)
        SaveMapPoints(f,mp); //TODO:SAVE MAP POINTS
    
    GetMapPointsIdx();

    std::cout << "The number of KeyFrames: " << mspKeyFrames.size() << endl;
    unsigned long int nKeyFrames = mspKeyFrames.size();
    f.write((char*)&nKeyFrames, sizeof(nKeyFrames));

    //保存关键帧KeyFrame
    for(auto kf:mspKeyFrames)
        SaveKeyFrame(f,kf);

    for(auto kf:mspKeyFrames)
    {
        //获取父节点，并保存父节点id
        KeyFrame* parent = kf->GetParent();
        unsigned long int parent_id = ULONG_MAX;
        if(parent)
            parent_id = parent->mnId;
        f.write((char*)&parent_id, sizeof(parent_id));
        
        unsigned long int nb_con = kf->GetConnectedKeyFrames().size();
        f.write((char*)&nb_con, sizeof(nb_con));

        for(auto ckf:kf->GetConnectedKeyFrames())
        {
            int weight = kf->GetWeight(ckf);
            bool mbDRKF = kf->isDRKF();
            f.write((char*)&ckf->mnId, sizeof(ckf->mnId));
            f.write((char*)&weight,sizeof(weight));
            f.write((char*)&mbDRKF, sizeof(mbDRKF));
        }
    }

    f.close();
    std::cout<< "Map Saving Finished!" <<std::endl;
}

void Map::SaveMapPoints(ofstream& f, MapPoint* mp)
{
    f.write((char*)&mp->mnId, sizeof(mp->mnId));
    cv::Mat mpWorldPos = mp->GetWorldPos();
    f.write((char*)& mpWorldPos.at<float>(0), sizeof(float));
    f.write((char*)& mpWorldPos.at<float>(1), sizeof(float));
    f.write((char*)& mpWorldPos.at<float>(2), sizeof(float));
}

void Map::GetMapPointsIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    unsigned long int i=0;
    for(auto mp:mspMapPoints)
    {
        mmpnMapPointsIdx[mp] = i;
        i+=1;
    }
}

void Map::SaveKeyFrame(ofstream &f, KeyFrame* kf)
{
    //保存当前关键帧的ID和时间戳
    bool mbDRKF = kf->isDRKF();
    f.write((char*)&kf->mnId, sizeof(kf->mnId));
    f.write((char*)&kf->mnFrameId, sizeof(kf->mnFrameId));
    f.write((char*)&kf->mTimeStamp, sizeof(kf->mTimeStamp));
    f.write((char*)&mbDRKF,sizeof(mbDRKF)); //shinan new added to tell whether it's a KF created in DR MODE
    f.write((char*)&kf->mbInitKF, sizeof(kf->mbInitKF));
    cv::Mat Tcw = kf->GetPose();
    std::vector<float> Quat = Converter::toQuaternion(Tcw);
    for (int i=0; i<4;i++)
    {
        f.write((char*)&Quat[i], sizeof(float));  //TODO:  to check whether here write an nan data
    }

    for(int i=0;i<3;i++)
    {
        f.write((char*)&Tcw.at<float>(i,3), sizeof(float));
    }
    
    //保存当前关键帧包含的ORB特征数目
    f.write((char*)& kf->N , sizeof(kf->N));
    for(int i =0; i< kf->N; i++)
    {
        //保存坐标，大小，角度，response，以及所属的octave response的值表明了特征点的goodness，这个值越高，说明该特征点越好
        cv::KeyPoint kp = kf->mvKeys[i];
        f.write((char*)&kp.pt.x, sizeof(kp.pt.x));
        f.write((char*)&kp.pt.y, sizeof(kp.pt.y));
        f.write((char*)&kp.size, sizeof(kp.size));
        f.write((char*)&kp.angle, sizeof(kp.angle));
        f.write((char*)&kp.response, sizeof(kp.response));
        f.write((char*)&kp.octave, sizeof(kp.octave));

        for(int j=0; j<kf->mDescriptors.cols;j++)
            f.write((char*)&kf->mDescriptors.at<uchar>(i,j),sizeof(uchar));
        unsigned long int mnIdx;
        MapPoint* mp = kf->GetMapPoint(i);
        if(mp==NULL)
            mnIdx = ULONG_MAX;
        else
        {
            mnIdx = mmpnMapPointsIdx[mp];
        }
        f.write((char*)&mnIdx, sizeof(mnIdx));  
    }
}

void Map::Load(const string &filename, SystemSetting* mySystemSetting,KeyFrameDatabase* mpKeyFrameDatabase)
{
    std::cout << "Map reading from binary file: "  << mySystemSetting << std::endl;
    ifstream f;
    f.open(filename.c_str());

    unsigned long int nMapPoints;
    f.read((char*)&nMapPoints, sizeof(nMapPoints));

    //依次读取每一个MapPoints， 并将其加入到地图中
    std::cout << "The number of MapPoints: " << nMapPoints << std::endl;
    for(unsigned int i = 0; i <nMapPoints;i++)
    {
        MapPoint* mp = LoadMapPoint(f);
        AddMapPoint(mp);
    }
    //获取所有的MapPoints
    std::vector<MapPoint*> vmp = GetAllMapPoints();
    //读取关键帧的数目
    unsigned long int nKeyFrames;
    f.read((char*)&nKeyFrames, sizeof(nKeyFrames));
    std::cout << "The number of KeyFrames:" << nKeyFrames <<endl;

   //一次读取关键帧，并加入到地图
    vector<KeyFrame*>kf_by_order;
    for(unsigned int i=0; i<nKeyFrames;i++)
    {
        KeyFrame* kf = LoadKeyFrame(f, mySystemSetting,mpKeyFrameDatabase);
        //shinan added to check whether kf pose is rightly loaded
        cout << "i is " << i << " kf id is " << kf->mnId << "with time "
             << setprecision(19) << kf->mTimeStamp * 1e6  
             << "with pose " << endl 
             << kf->GetPose() << endl;
        AddKeyFrame(kf);
        kf_by_order.push_back(kf);
        mpKeyFrameDatabase->add(kf);
    }

    if(mnMaxKFid >0 )  //保证新的帧在接下来的帧之后
    {
        Frame temp_frame = Frame(mnMaxKFid);
        KeyFrame::setKeyFrame(mnMaxKFid);
    }
    cerr<<">>>>>>>KeyFrame Load OVER!<<<<<<<"<<endl;
     //读取生长树；
    map<unsigned long int, KeyFrame*> kf_by_id;
    for ( auto kf: mspKeyFrames ){
        kf_by_id[kf->mnId] = kf;
    }   

    cerr<<"Start Load The Parent!"<<endl;
    for( auto kf: kf_by_order )
    {
        //std::cout << "----Loading frame " << kf->mnId<<endl;
        //读取当前关键帧的父节点ID；
        unsigned long int parent_id;
        f.read((char*)&parent_id, sizeof(parent_id));

        //给当前关键帧添加父节点关键帧；
        if ( parent_id != ULONG_MAX )
            kf->ChangeParent(kf_by_id[parent_id]);

        //读取当前关键帧的关联关系；
        //先读取当前关键帧的关联关键帧的数目；
        unsigned long int nb_con;
        f.read((char*)&nb_con, sizeof(nb_con));
        //然后读取每一个关联关键帧的ID和weight，并把该关联关键帧加入关系图中；
        for ( unsigned long int i = 0; i < nb_con; i ++ )
        {
            unsigned long int id;
            int weight;
            bool mbDRKF;
            f.read((char*)&id, sizeof(id));
            f.read((char*)&weight, sizeof(weight));
            f.read((char*)&mbDRKF,sizeof(mbDRKF));
            //std::cout << "------got connected frame id is " << id  << "its mbDRKF is " << mbDRKF << endl;
            if(kf_by_id[id]==nullptr)
            {
                std::cout << " covisibility kfid " << id << " does not be saved as KEYFRAME its weight is "<< weight<<endl;
                continue;
            }
            kf->AddConnection(kf_by_id[id],weight);
        }
    }
    cerr<<"Parent Load OVER!"<<endl;
    for ( auto mp: vmp )
    {
        if(mp)
        {
             mp->ComputeDistinctiveDescriptors();
             mp->UpdateNormalAndDepth();
         }
    }

    f.close();
    cerr<<"Load IS OVER!"<<endl;
    return;

}

KeyFrame* Map::LoadKeyFrame(ifstream &f, SystemSetting*  mySystemSetting,KeyFrameDatabase* mpKeyFrameDatabase)
{
    InitKeyFrame initkf(*mySystemSetting);
    //读取id 和 时间戳
    f.read((char*)&initkf.nId , sizeof(initkf.nId));
    f.read((char*)&initkf.mnFrameId,sizeof(initkf.mnFrameId));
    f.read((char*)&initkf.TimeStamp, sizeof(initkf.TimeStamp));
    f.read((char*)&initkf.mbDRKF, sizeof(initkf.mbDRKF));
    f.read((char*)&initkf.mbInitKF, sizeof(initkf.mbInitKF));
    //cout << "LOADED KF "<< initkf.nId << "with DRKF valueed " << initkf.mbDRKF<< endl; 
    //读取Tcw
    cv::Mat T = cv::Mat::zeros(4,4,CV_32F);
    std::vector<float> Quat(4);

    for(int i=0;i<4;i++)
        f.read((char*)&Quat[i], sizeof(float));
    for(int i=0;i<4;i++)  //added in case that the value inside is too small
    {
        if(abs(Quat[i]) < 1e-7)
            Quat[i] = 0.0;
    }
    cv::Mat Rcw = Converter::QuaterniontoMat(Quat);
    for(int i=0; i<3;i++)
        f.read((char*)&T.at<float>(i,3),sizeof(float));
    for(int i=0;i<3;i++)  //added in case that the value inside is too small
    {
        if(abs(T.at<float>(i,3)) < 1e-7)
            T.at<float>(i,3) = 0.0;
    }
    Rcw.copyTo(T.colRange(0,3).rowRange(0,3));
    T.at<float>(3,3) = 1;
    //读取当前关键帧中特征点的数目
    f.read((char*)&initkf.N, sizeof(initkf.N));
    initkf.vKps.reserve(initkf.N);
    initkf.Descriptors.create(initkf.N, 32, CV_8UC1);
    vector<float>KeypointDepth;
 
    std::vector<MapPoint*> vpMapPoints;
    vpMapPoints = vector<MapPoint*>(initkf.N,static_cast<MapPoint*>(NULL));
    //依次读取当前关键帧的特征点和描述符；
    std::vector<MapPoint*> vmp = GetAllMapPoints();
    for(int i = 0; i < initkf.N; i ++ )
    {
        cv::KeyPoint kp;
        f.read((char*)&kp.pt.x, sizeof(kp.pt.x));
        f.read((char*)&kp.pt.y, sizeof(kp.pt.y));
        f.read((char*)&kp.size, sizeof(kp.size));
        f.read((char*)&kp.angle,sizeof(kp.angle));
        f.read((char*)&kp.response, sizeof(kp.response));
        f.read((char*)&kp.octave, sizeof(kp.octave));

        initkf.vKps.push_back(kp);
        //读取当前特征点的描述符；
        for ( int j = 0; j < 32; j ++ ) //TODO: 修改为initkf.Descriptors.cols
            f.read((char*)&initkf.Descriptors.at<unsigned char>(i,j),sizeof(char));

        //读取当前特征点和MapPoints的对应关系；
        unsigned long int mpidx;
        f.read((char*)&mpidx, sizeof(mpidx));

        //从vmp这个所有的MapPoints中查找当前关键帧的MapPoint，并插入
        if( mpidx == ULONG_MAX )
            vpMapPoints[i] = NULL;
        else
            vpMapPoints[i] = vmp[mpidx];
    }

    initkf.vRight = vector<float>(initkf.N,-1);
    initkf.vDepth = vector<float>(initkf.N,-1);
    //initkf.vDepth = KeypointDepth;
    initkf.UndistortKeyPoints();
    initkf.AssignFeaturesToGrid();

    //使用initkf初始化一个关键帧，并设置相关参数
    KeyFrame* kf = new KeyFrame( initkf, this, mpKeyFrameDatabase, vpMapPoints );
    kf->mnId = initkf.nId;
    if(kf->mbInitKF == true) mvpKeyFrameOrigins.push_back(kf);
    kf->SetPose(T);
    kf->ComputeBoW();
    for ( int i = 0; i < initkf.N; i ++ )
    {
        if ( vpMapPoints[i] )
        {
            vpMapPoints[i]->AddObservation(kf,i);
            if( !vpMapPoints[i]->GetReferenceKeyFrame())
                vpMapPoints[i]->SetReferenceKeyFrame(kf);
        }
    }
    return kf;
}


MapPoint* Map::LoadMapPoint( ifstream &f )
{
    //主要包括MapPoints的位姿和ID；
    cv::Mat Position(3,1,CV_32F);
    long unsigned int id;
    f.read((char*)&id, sizeof(id));
    f.read((char*)&Position.at<float>(0), sizeof(float));
    f.read((char*)&Position.at<float>(1), sizeof(float));
    f.read((char*)&Position.at<float>(2), sizeof(float));
    //初始化一个MapPoint，并设置其ID和Position；
    MapPoint* mp = new MapPoint(Position, this );
    mp->mnId = id;
    mp->SetWorldPos( Position );
    return mp;
}

}  // namespace ORB_SLAM2
