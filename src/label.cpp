#include "try.h"

using namespace std;
using namespace cv;

//³¬ÏñËØÌî³ä
void pixelCluster(const vector<vector<Point2i>> &Coordinate, const vector<int> &numOfEachSuperpixel, const Mat &Label2, const int superPixelNum, Mat & seg)
{
	for (int i = 0; i < superPixelNum; i++)
	{
		int temp = Label2.at<int>(i, 0);

		   
			switch (temp / 10)
			{
			case 0:
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x;
					int y = Coordinate[i][j].y;
					seg.at<Vec3b>(x, y)[0] = temp * 20;
					seg.at<Vec3b>(x, y)[1] = 255;
					seg.at<Vec3b>(x, y)[2] = 255;
				}
				break;
			case 1:
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x;
					int y = Coordinate[i][j].y;
					seg.at<Vec3b>(x, y)[0] = (temp - 10) * 20;
					seg.at<Vec3b>(x, y)[1] = (temp - 10) * 20;
					seg.at<Vec3b>(x, y)[2] = 255;
				}
				break;
			case 2:
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x;
					int y = Coordinate[i][j].y;
					seg.at<Vec3b>(x, y)[0] = (temp - 20) * 20;
					seg.at<Vec3b>(x, y)[1] = (temp - 20) * 20;
					seg.at<Vec3b>(x, y)[2] = (temp - 20) * 20;
				}
				break;
			case 3:
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x;
					int y = Coordinate[i][j].y;
					seg.at<Vec3b>(x, y)[0] = (temp - 30) * 20;
					seg.at<Vec3b>(x, y)[1] = 200;
					seg.at<Vec3b>(x, y)[2] = 200;
				}
				break;
			case 4:
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x;
					int y = Coordinate[i][j].y;
					seg.at<Vec3b>(x, y)[0] = (temp - 40) * 20;
					seg.at<Vec3b>(x, y)[1] = (temp - 40) * 20;
					seg.at<Vec3b>(x, y)[2] = 200;
				}
				break;
			}
	}
}