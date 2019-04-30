#ifndef _GRADUATIAON_H
#define _GRADUATIAON_H

#include <opencv2/opencv.hpp>




using namespace std;
using namespace cv;

//-----------------------序列图像读取------------------------------

int readImageSequence(const char* path, const char* postfix, vector<Mat> &mat);

//-----------------------Shi-Tomasi角点检测--------------------------
void cornerDetection(const Mat& mat, vector<Point2f> &firstPoint);

//---------------------LK金字塔光流生成点轨迹矩阵---------------------
void trajectoryGeneration(const vector<Mat> &srcGrayImage, const vector<Point2f> &firstPoint, const int range, vector<vector<Point2f> > &rawPoint, Mat &W);

//---------------------存储追踪成功的角点---------------------
void storePoint(const Mat &W, const int range, vector<vector<Point2f> > &Point);

//-------------------------SSC聚类----------------------------

//正则化参数
double computeLambda(const Mat & W);

//仿射空间系数矩阵
Mat admmLasso(const Mat& W, const double Lambda1);

//thrC
Mat thrC(const Mat & C2);

//BuildAdjacency
Mat buildAdjacency(const Mat& cP);

//SpectralClustering
Mat spectralClustering(const Mat &CKSym, const int n);

//确定前背景标记
void label(const Mat &Label, uchar &foreG, uchar &backG);

//----------------------------------ORB追踪---------------------------------

//计算给定点集横纵坐标极值
void computeCor(const vector<Point2f> &Point, float &maxX, float &maxY, float &minX, float &minY);

//计算图像序列特征点和区域
void track(const vector<Mat> &srcGrayImage, vector<vector<Point2f> > &featurePoint, vector<vector<Point2f> > &orbPoint, vector<vector<float> > &featureCor, vector<vector<float> > &orbCor);

//---------------------超像素--------------------

void fillMat(Mat& L, int* labels, int dimx, int dimy);

void getBoundaryImage(Mat& B, Mat& L, int dimx, int dimy);

void getOverlayedImage(Mat& R, Mat& B, Mat& I);

void computeCoordinate(const Mat & L, const int superPixelNum, const int dimx, const int dimy,  vector <vector<Point2i>> &Coordinate);

//---------------------------提取颜色特征--------------

//计算HSV超像素容器
void buildHsvSuperPiexlVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &NormIwld, const int superPixelNum, vector<Mat> & textureFeatureVector);

//计算颜色特征向量
void ColorFeature(const vector<Mat> &hsvSuperPixelVector, const int hueBinNum, const int saturationBinNum, const int valueBinNum, const int superPixelNum, vector<Mat> &colorFeature);

//---------------------------提取纹理特征--------------

//计算Iwld值
void computeIwld(const Mat &gray, const int dimx, const int dimy, Mat &IwldValue);

//计算标准化Iwld超像素容器
void buildIwldVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &NormIwld, const int superPixelNum, vector<Mat> & textureFeatureVector);

//计算纹理特征向量
void TextureFeature(const vector<Mat> &textureFeatureVector, const int superPixelNum, vector<Mat> &textureFeature);
//---------------------------超像素聚类--------------

//计算颜色空间距离
void computeColorDis(vector<Mat> &colorFeature, const int superPixelNum, Mat & colorDistance, double & sumOfColDis);

//计算纹理空间距离
void computeTexDis(vector<Mat> &textureFeature, const int superPixelNum, Mat & textureDistance, double & sumOfTexDis);

//计算曼哈顿距离
void computeManDis(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const int superPixelNum, Mat & manDistance);

//计算权重矩阵
void computeWeight(Mat & superPixelDis, const int superPixelNum, const double Xi, Mat & Weight);

//计算度矩阵D
void computeD(Mat & Weight, const int superPixelNum, Mat & DN);

//计算Y矩阵
void computeV(Mat & V, const int numOfCluster, const int superPixelNum, Mat & vtY);

//---------------------------图像标记--------------

void pixelCluster(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const Mat &Label2, const int superPixelNum, Mat & seg);

#endif
