#ifndef _GRADUATIAON_H
#define _GRADUATIAON_H

#include <opencv2/opencv.hpp>




using namespace std;
using namespace cv;

//-----------------------����ͼ���ȡ------------------------------

int readImageSequence(const char* path, const char* postfix, vector<Mat> &mat);

//-----------------------Shi-Tomasi�ǵ���--------------------------
void cornerDetection(const Mat& mat, vector<Point2f> &firstPoint);

//---------------------LK�������������ɵ�켣����---------------------
void trajectoryGeneration(const vector<Mat> &srcGrayImage, const vector<Point2f> &firstPoint, const int range, vector<vector<Point2f> > &rawPoint, Mat &W);

//---------------------�洢׷�ٳɹ��Ľǵ�---------------------
void storePoint(const Mat &W, const int range, vector<vector<Point2f> > &Point);

//-------------------------SSC����----------------------------

//���򻯲���
double computeLambda(const Mat & W);

//����ռ�ϵ������
Mat admmLasso(const Mat& W, const double Lambda1);

//thrC
Mat thrC(const Mat & C2);

//BuildAdjacency
Mat buildAdjacency(const Mat& cP);

//SpectralClustering
Mat spectralClustering(const Mat &CKSym, const int n);

//ȷ��ǰ�������
void label(const Mat &Label, uchar &foreG, uchar &backG);

//----------------------------------ORB׷��---------------------------------

//��������㼯�������꼫ֵ
void computeCor(const vector<Point2f> &Point, float &maxX, float &maxY, float &minX, float &minY);

//����ͼ�����������������
void track(const vector<Mat> &srcGrayImage, vector<vector<Point2f> > &featurePoint, vector<vector<Point2f> > &orbPoint, vector<vector<float> > &featureCor, vector<vector<float> > &orbCor);

//---------------------������--------------------

void fillMat(Mat& L, int* labels, int dimx, int dimy);

void getBoundaryImage(Mat& B, Mat& L, int dimx, int dimy);

void getOverlayedImage(Mat& R, Mat& B, Mat& I);

void computeCoordinate(const Mat & L, const int superPixelNum, const int dimx, const int dimy,  vector <vector<Point2i>> &Coordinate);

//---------------------------��ȡ��ɫ����--------------

//����HSV����������
void buildHsvSuperPiexlVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &NormIwld, const int superPixelNum, vector<Mat> & textureFeatureVector);

//������ɫ��������
void ColorFeature(const vector<Mat> &hsvSuperPixelVector, const int hueBinNum, const int saturationBinNum, const int valueBinNum, const int superPixelNum, vector<Mat> &colorFeature);

//---------------------------��ȡ��������--------------

//����Iwldֵ
void computeIwld(const Mat &gray, const int dimx, const int dimy, Mat &IwldValue);

//�����׼��Iwld����������
void buildIwldVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &NormIwld, const int superPixelNum, vector<Mat> & textureFeatureVector);

//����������������
void TextureFeature(const vector<Mat> &textureFeatureVector, const int superPixelNum, vector<Mat> &textureFeature);
//---------------------------�����ؾ���--------------

//������ɫ�ռ����
void computeColorDis(vector<Mat> &colorFeature, const int superPixelNum, Mat & colorDistance, double & sumOfColDis);

//��������ռ����
void computeTexDis(vector<Mat> &textureFeature, const int superPixelNum, Mat & textureDistance, double & sumOfTexDis);

//���������پ���
void computeManDis(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const int superPixelNum, Mat & manDistance);

//����Ȩ�ؾ���
void computeWeight(Mat & superPixelDis, const int superPixelNum, const double Xi, Mat & Weight);

//����Ⱦ���D
void computeD(Mat & Weight, const int superPixelNum, Mat & DN);

//����Y����
void computeV(Mat & V, const int numOfCluster, const int superPixelNum, Mat & vtY);

//---------------------------ͼ����--------------

void pixelCluster(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const Mat &Label2, const int superPixelNum, Mat & seg);

#endif
