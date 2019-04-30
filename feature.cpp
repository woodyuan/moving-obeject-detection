#include "try.h"
#include <math.h>
#define _USE_MATH_DEFINES

#define pi 3.1416

using namespace std;
using namespace cv;

//计算HSV超像素容器
void buildHsvSuperPiexlVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &hsvImage,const int superPixelNum, vector<Mat> & hsvSuperPixelVector)
{
	for (int i = 0; i < superPixelNum; i++)
	{
		Mat temp;
		for (int j = 0; j < numOfEachSuperpixel[i]; j++)
		{
			int row = Coordinate[i][j].x;
			int col = Coordinate[i][j].y;
			temp.push_back(hsvImage.at<Vec3b>(row, col));
		}
		hsvSuperPixelVector.push_back(temp.clone());
	}
}

//计算颜色特征向量
void ColorFeature(const vector<Mat> &hsvSuperPixelVector, const int hueBinNum, const int saturationBinNum, const int valueBinNum, const int superPixelNum, vector<Mat> &colorFeature)
{
	int histSize[] = { hueBinNum,saturationBinNum,valueBinNum };
	float hueRange[] = { 0,180 };
	float saturationRange[] = { 0,256 };
	float valueRange[] = { 0,256 };

	const float* range[] = { hueRange,saturationRange,valueRange };
	int channel[] = { 0,1,2 };

	for (int i = 0; i < superPixelNum; i++)
	{
		MatND temp;
		int n = 0;
		Mat dsthist = Mat::zeros(hueBinNum*saturationBinNum*valueBinNum, 1, CV_32F);
		calcHist(&hsvSuperPixelVector[i], 1, channel, MatND(), temp, 3, histSize, range, true, false);

		for (int j = 0; j < hueBinNum; j++)
			for (int l = 0; l < saturationBinNum; l++)
				for (int k = 0; k < valueBinNum; k++)
				{
					dsthist.at<float>(n, 0) = temp.at<float>(j, l, k);
					n++;
				}
		colorFeature.push_back(dsthist);
	}
}

//---------------------------提取纹理特征--------------

//计算Iwld值
void computeIwld(const Mat &gray, const int dimx, const int dimy,Mat &IwldValue)
{
	if (gray.at<uchar>(0, 0) != 0)
		IwldValue.at<double>(0, 0) = atan((double)(gray.at<uchar>(0, 1) + gray.at<uchar>(1, 0) + gray.at<uchar>(1, 1)) * 3 / gray.at<uchar>(0, 0) - 9);
	else
		IwldValue.at<double>(0, 0) = pi / 2;

	if (gray.at<uchar>(dimy - 1, 0) != 0)
		IwldValue.at<double>(dimy - 1, 0) = atan((double)(gray.at<uchar>(dimy - 2, 0) + gray.at<uchar>(dimy - 2, 1) + gray.at<uchar>(dimy - 1, 1)) * 3 / gray.at<uchar>(dimy - 1, 0) - 9);
	else
		IwldValue.at<double>(dimy - 1, 0) = pi / 2;

	if (gray.at<uchar>(0, dimx - 1) != 0)
		IwldValue.at<double>(0, dimx - 1) = atan((double)(gray.at<uchar>(0, dimx - 2) + gray.at<uchar>(1, dimx - 2) + gray.at<uchar>(1, dimx - 1)) * 3 / gray.at<uchar>(0, dimx - 1) - 9);
	else
		IwldValue.at<double>(0, dimx - 1) = pi / 2;

	if (gray.at<uchar>(dimy - 1, dimx - 1) != 0)
		IwldValue.at<double>(dimy - 1, dimx - 1) = atan((double)(gray.at<uchar>(dimy - 2, dimx - 2) + gray.at<uchar>(dimy - 1, dimx - 2) + gray.at<uchar>(dimy - 2, dimx - 1)) * 3 / gray.at<uchar>(dimy - 1, dimx - 1) - 9);
	else
		IwldValue.at<double>(dimy - 1, dimx - 1) = pi / 2;

	for (int i = 1; i < dimy - 1; i++)
		for (int j = 1; j < dimx - 1; j++)
		{
			if (gray.at<uchar>(i, j) != 0)
				IwldValue.at<double>(i, j) = atan((double)(gray.at<uchar>(i - 1, j - 1) + gray.at<uchar>(i, j - 1) + gray.at<uchar>(i + 1, j - 1) +
					gray.at<uchar>(i - 1, j) + gray.at<uchar>(i + 1, j) + gray.at<uchar>(i - 1, j + 1) +
					gray.at<uchar>(i, j + 1) + gray.at<uchar>(i + 1, j + 1)) * 3 / gray.at<uchar>(i, j) - 24);
			else
				IwldValue.at<double>(i, j) = pi / 2;
		}

	for (int j = 1; j < dimx - 1; j++)
	{
		if (gray.at<uchar>(0, j) != 0)
			IwldValue.at<double>(0, j) = atan((double)(gray.at<uchar>(0, j - 1) + gray.at<uchar>(1, j - 1) + gray.at<uchar>(1, j) +
				gray.at<uchar>(0, j + 1) + gray.at<uchar>(1, j + 1)) * 3 / gray.at<uchar>(0, j) - 15);
		else
			IwldValue.at<double>(0, j) = pi / 2;

		if (gray.at<uchar>(dimy - 1, j) != 0)
			IwldValue.at<double>(dimy - 1, j) = atan((double)(gray.at<uchar>(dimy - 2, j - 1) + gray.at<uchar>(dimy - 1, j - 1) + gray.at<uchar>(dimy - 2, j) +
				gray.at<uchar>(dimy - 2, j + 1) + gray.at<uchar>(dimy - 1, j + 1)) * 3 / gray.at<uchar>(dimy - 1, j) - 15);
		else
			IwldValue.at<double>(dimy - 1, j) = pi / 2;
	}

	for (int i = 1; i < dimy - 1; i++)
	{
		if (gray.at<uchar>(i, 0) != 0)
			IwldValue.at<double>(i, 0) = atan((double)(gray.at<uchar>(i - 1, 0) + gray.at<uchar>(i - 1, 1) + gray.at<uchar>(i, 1) + gray.at<uchar>(i + 1, 0) +
				gray.at<uchar>(i + 1, 1)) * 3 / gray.at<uchar>(i, 0) - 15);
		else
			IwldValue.at<double>(i, 0) = pi / 2;

		if (gray.at<uchar>(i, dimx - 1) != 0)
			IwldValue.at<double>(i, dimx - 1) = atan((double)(gray.at<uchar>(i - 1, dimx - 1) + gray.at<uchar>(i - 1, dimx - 2) + gray.at<uchar>(i, dimx - 2) + gray.at<uchar>(i + 1, dimx - 1) +
				gray.at<uchar>(i + 1, dimx - 2)) * 3 / gray.at<uchar>(i, dimx - 1) - 15);
		else
			IwldValue.at<double>(i, dimx - 1) = pi / 2;
	}
}

//计算标准化Iwld超像素容器
void buildIwldVector(const vector<vector<Point2i> > & Coordinate, const vector<int> & numOfEachSuperpixel, const Mat &NormIwld, const int superPixelNum, vector<Mat> & textureFeatureVector)
{
	for (int k = 0; k < superPixelNum; k++)
	{
		Mat temp;
		for (int j = 0; j < numOfEachSuperpixel[k]; j++)
		{

			int row = Coordinate[k][j].x;
			int col = Coordinate[k][j].y;
			temp.push_back(NormIwld.at<uchar>(row, col));
		}
		textureFeatureVector.push_back(temp.clone());
	}

//计算纹理特征向量
}
void TextureFeature(const vector<Mat> &textureFeatureVector,const int superPixelNum, vector<Mat> &textureFeature)
{
	int histSize[] = { 256 };
	float ranges[] = { 0, 256 };
	const float* range[] = { ranges };
	int channel[] = { 0 };

	for (int i = 0; i < superPixelNum; i++)
	{
		MatND temp;
		Mat dsthist = Mat::zeros(256, 1, CV_32F);
		calcHist(&textureFeatureVector[i], 1, channel, MatND(), temp, 1, histSize, range, true, false);

		for (int j = 0; j < 256; j++)
		{
			dsthist.at<float>(j, 0) = temp.at<float>(j, 0);
		}
		textureFeature.push_back(dsthist);
	}
}


