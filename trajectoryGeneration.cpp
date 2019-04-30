#include "try.h"

using namespace std;
using namespace cv;



//---------------------序列图像读取-------------------------

int readImageSequence(const char* path, const char* postfix, vector<Mat> &mat)
{
	Mat frame;									//读取的图像帧
	char inFileName[100] = { 0 };				//存储文件名
	int i = 0;

	//将图像序列存入Vector中

	for (;; i++)
	{
		sprintf_s(inFileName, 100, "%s%d%s", path, i + 1, postfix);
		Mat frame = imread(inFileName);
		if (frame.empty())
			break;
		mat.push_back(frame.clone());
	}
	return i;
}

//-----------------------Shi-Tomasi角点检测--------------------------

void cornerDetection(const Mat& mat, vector<Point2f> &firstPoint)
{
	int winWidth = 80;				//检测角点窗口的宽度
	int winHeight = 120;			//检测角点窗口的高度
	int maxCornerNumber = 10;		//区域内角点最大数量
	double qualityLevel = 0.01;		//角点检测可接受的最小特征值
	double minDistance = 10;		//角点之间的最小距离
	int blockSize = 3;				//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;				//权重系数

	vector<Point2f> corner;					    //存储检测到的角点
	int colNumOfWin = mat.cols / winWidth;		//窗口的列数
	int rawNumOfWin = mat.rows / winHeight;	    //窗口的行数

	//Shi-Tomasi角点检测

	for (int j = 0; j < rawNumOfWin; j++)
	{
		int yCoordinate = j * winHeight;
		for (int i = 0; i < colNumOfWin; i++)
		{
			int xCoordinate = i * winWidth;

			//将原图像均分为N个区域
			Rect rect(xCoordinate, yCoordinate, winWidth, winHeight);
			Mat roi(mat, rect);

			//获取角点
			goodFeaturesToTrack(roi, corner, maxCornerNumber, qualityLevel, minDistance, Mat(), blockSize, false, k);

			//将每一个区域的角点整合
			for (int k = 0; k < corner.size(); k++)
			{
				Point2f tempPoint;
				tempPoint.x = corner[k].x + xCoordinate;
				tempPoint.y = corner[k].y + yCoordinate;
				firstPoint.push_back(tempPoint);
			}
		}
	}
}

//---------------------LK金字塔光流生成点轨迹矩阵---------------------

void trajectoryGeneration(const vector<Mat> &srcGrayImage, const vector<Point2f> &firstPoint,const int range, vector<vector<Point2f> > &rawPoint, Mat &W)
{
	int maxLevel = 3;
	int Flag = 0;
	double minEigThreshold = 0.001;
	size_t sP = firstPoint.size();
	Size winSize(9, 9);
	TermCriteria Termcrit(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01);
	vector<uchar> status;
	vector<float> error;
	vector<uchar> preStatus(sP);

	fill(preStatus.begin(), preStatus.end(), 1);
	rawPoint[0] = firstPoint;
	
	for (size_t i = 0; i < range - 1; i++)
	{
		vector<Point2f> temp;

		rawPoint[i + 1].resize(sP);
		fill(rawPoint[i + 1].begin(), rawPoint[i + 1].end(), Point2f(-100, -100));

		calcOpticalFlowPyrLK(srcGrayImage[i], srcGrayImage[i + 1], rawPoint[i], temp, status, error, winSize, maxLevel, Termcrit, Flag, minEigThreshold);

		for (size_t j = 0; j < sP; j++)
		{
			if (preStatus[j] == 0)
				status[j] = 0;
			else if (temp[j].x < 0 || temp[j].x > srcGrayImage[i+1].cols || temp[j].y < 0 || temp[j].y > srcGrayImage[i+1].rows)
				status[j] = 0;
		}

		for (size_t k = 0; k < sP; k++)
		{
			if (status[k] == 1)
				rawPoint[i + 1][k] = temp[k];
			
		}
		preStatus = status;
	}

	//角点轨迹矩阵，每一列代表一个角点，且已删除追踪失败的角点

	Mat Y(2 * range, (int)sP, CV_64F);
	int jj = 0;
	int ii = 0;
	int count = 0;
	for (size_t j = 0; j < sP; j++)
	{
		if (status[j] == 0)	//判定是否追踪失败，失败则不添加进矩阵
			continue;
		for (size_t i = 0; i < range; i++)
		{
			Y.at<double>(ii, jj) = cvRound(rawPoint[i][j].x);
			Y.at<double>(ii + 1, jj) = cvRound(rawPoint[i][j].y);
			ii = ii + 2;
		}
		jj++;
		ii = 0;
		count++;
	}

	//重新分配轨迹矩阵大小

	Rect rect(0, 0, count, 2 * range);
	Mat temp(Y, rect);
	W.push_back(temp.clone());
}

//-----------------------存储追踪成功的角点--------------------------

void storePoint(const Mat &W, const int range,vector<vector<Point2f> > &Point)
{
	int col = W.cols;
	int row = W.rows;
	for (int j = 0; j < col; j++)
	{
		int ii = 0;
		for (int i = 0; i < row; i = i + 2)
		{
			Point2f temp;
			temp.x = (float)W.at<double>(i, j);
			temp.y = (float)W.at<double>(i + 1, j);
			Point[ii].push_back(temp);
			ii++;
		}
	}
}


