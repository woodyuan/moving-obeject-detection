#include <opencv2/opencv.hpp>
#include "preemptiveSLIC.h"

using namespace std;
using namespace cv;

void fillMat(Mat& L, int* labels, int dimx, int dimy)
{
	// label image L
	L = Mat(dimy, dimx, CV_32SC1);
	for (int i = 0; i<dimy; i++)
	{
		for (int j = 0; j<dimx; j++)
		{
			int idx = dimx*i + j; // row major order
			L.at<int>(i, j) = labels[idx];
		}
	}
}

void getBoundaryImage(Mat& B, Mat& L, int dimx, int dimy)
{
	// boundary image B
	B = Mat::zeros(dimy, dimx, CV_8UC1);
	for (int i = 1; i<dimy - 1; i++)
	{
		for (int j = 1; j<dimx - 1; j++)
		{
			if (L.at<int>(i, j) != L.at<int>(i + 1, j) || L.at<int>(i, j) != L.at<int>(i, j + 1))
				B.at<uchar>(i, j) = 1;
		}
	}
}

void getOverlayedImage(Mat& R, Mat& B, Mat& I)
{
	int dimx = I.cols;
	int dimy = I.rows;

	// overlayed image  
	I.copyTo(R);
	for (int i = 1; i<dimy - 1; i++)
	{
		for (int j = 1; j<dimx - 1; j++)
		{
			if (B.at<uchar>(i, j))
			{
				R.at<cv::Vec3b>(i, j)[0] = 255;
				R.at<cv::Vec3b>(i, j)[1] = 255;
				R.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
}

//¼ÆËã³¬ÏñËØ×ø±ê
void computeCoordinate(const Mat & L, const int superPixelNum, const int dimx, const int dimy, vector <vector<Point2i>> &Coordinate)
{
	for (int k = 0; k < superPixelNum; k++)
	{
		Mat subtraction(dimy, dimx, CV_32S, Scalar::all(k));
		subtraction -= L;
		subtraction = abs(subtraction);
		Mat Ind, Order;
		cv::sort(subtraction, Order, SORT_EVERY_COLUMN + SORT_ASCENDING);
		sortIdx(subtraction, Ind, SORT_EVERY_COLUMN + SORT_ASCENDING);
		Mat temp;
		for (int j = 0; j < dimx; j++)
		{
			if (Order.at<float>(0, j) == 0)
			{
				for (int i = 0; i < dimy; i++)
				{
					if (Order.at<float>(i, j) == 0)
					{
						Point2i temp;
						int loc = Ind.at<int>(i, j);
						temp.x = loc;
						temp.y = j;
						Coordinate[k].push_back(temp);
					}
					else
						break;
				}
			}
			else
				continue;
		}
	}
}