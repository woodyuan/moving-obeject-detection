#include "try.h"
#include <cmath>

using namespace std;
using namespace cv;

//---------------------------≥¨œÒÀÿæ€¿‡--------------

//º∆À„—’…´ø’º‰æ‡¿Î
void computeColorDis(vector<Mat> &colorFeature, const int superPixelNum,  Mat & colorDistance, double & sumOfColDis)
{
	double correlation = 0;
	Mat temp;
	for (int i = 0; i < superPixelNum; i++)
		for (int j = i; j < superPixelNum; j++)
		{
			correlation = compareHist(colorFeature[i], colorFeature[j], HISTCMP_CORREL);
			colorDistance.at<double>(i, j) = (double)(1 - correlation) * 0.5;
			sumOfColDis += colorDistance.at<double>(i, j);
		}
	temp = colorDistance.t();
	colorDistance = colorDistance + temp;
}

//º∆À„Œ∆¿Ìø’º‰æ‡¿Î
void computeTexDis(vector<Mat> &textureFeature, const int superPixelNum, Mat & textureDistance, double & sumOfTexDis)
{
	double correlation = 0;
	Mat temp;
	for (int i = 0; i < superPixelNum; i++)
		for (int j = i; j < superPixelNum; j++)
		{
			correlation = compareHist(textureFeature[i], textureFeature[j], HISTCMP_CORREL);
			textureDistance.at<double>(i, j) = (double)(1 - correlation) * 0.5;
			sumOfTexDis += textureDistance.at<double>(i, j);
		}
	temp = textureDistance.t();
	textureDistance = textureDistance + temp;
}

//º∆À„¬¸π˛∂Ÿæ‡¿Î
void computeManDis(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const int superPixelNum,  Mat & manDistance)
{
	Mat temp;
	for (int i = 0; i < superPixelNum; i++)
	{
		int icenter = cvRound(numOfEachSuperpixel[i] / 2);
		int ilocX = Coordinate[i][icenter].x;
		int ilocY = Coordinate[i][icenter].y;
		for (int j = i; j < superPixelNum; j++)
		{
			int jcenter = cvRound(numOfEachSuperpixel[j] / 2);
			int jlocX = Coordinate[j][jcenter].x;
			int jlocY = Coordinate[j][jcenter].y;
			int dX = abs(ilocX - jlocX);
			int dY = abs(ilocY - jlocY);
			manDistance.at<double>(i, j) = dX + dY;
		}
	}
	temp = manDistance.t();
	manDistance = temp + manDistance;
}

//º∆À„»®÷ÿæÿ’Û
void computeWeight(Mat & superPixelDis, const int superPixelNum, const double Xi,Mat & Weight)
{
	double temp;
	Mat temp1;
	for (int i = 0; i < superPixelNum; i++)
		for (int j = i; j < superPixelNum; j++)
		{
			if (j == i)
				Weight.at<double>(i, i) = 0;
			else
			{
				temp = superPixelDis.at<double>(i, j);
				temp = -pow(superPixelDis.at<double>(i, j), 2) / pow(Xi, 2) * 0.5;
				Weight.at<double>(i, j) = exp(temp);
			}
		}
	temp1 = Weight.t();
	Weight = Weight + temp1;
}

//º∆À„∂»æÿ’ÛD
void computeD(Mat & Weight, const int superPixelNum, Mat & DN)
{
	Mat rSum(1, superPixelNum, CV_64F, Scalar(0));
	reduce(Weight, rSum, 0, REDUCE_SUM);
	for (int j = 0; j < superPixelNum; j++)
	{
		double temp = sqrt(rSum.at<double>(0, j));
		DN.at<double>(j, j) = 1 / temp;
	}
}

//º∆À„Yæÿ’Û
void computeV(Mat & V, const int numOfCluster, const int superPixelNum, Mat & vtY)
{
	for (int i = 0; i < superPixelNum; i++)
	{
		double temp;
		temp = norm(V.row(i), NORM_L2);
		for (int j = 0; j < numOfCluster; j++)
			vtY.at<float>(i, j) = V.at<double>(i, j) / temp;
	}
}

