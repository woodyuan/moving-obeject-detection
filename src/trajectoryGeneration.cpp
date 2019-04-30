#include "try.h"

using namespace std;
using namespace cv;



//---------------------����ͼ���ȡ-------------------------

int readImageSequence(const char* path, const char* postfix, vector<Mat> &mat)
{
	Mat frame;									//��ȡ��ͼ��֡
	char inFileName[100] = { 0 };				//�洢�ļ���
	int i = 0;

	//��ͼ�����д���Vector��

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

//-----------------------Shi-Tomasi�ǵ���--------------------------

void cornerDetection(const Mat& mat, vector<Point2f> &firstPoint)
{
	int winWidth = 80;				//���ǵ㴰�ڵĿ��
	int winHeight = 120;			//���ǵ㴰�ڵĸ߶�
	int maxCornerNumber = 10;		//�����ڽǵ��������
	double qualityLevel = 0.01;		//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 10;		//�ǵ�֮�����С����
	int blockSize = 3;				//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;				//Ȩ��ϵ��

	vector<Point2f> corner;					    //�洢��⵽�Ľǵ�
	int colNumOfWin = mat.cols / winWidth;		//���ڵ�����
	int rawNumOfWin = mat.rows / winHeight;	    //���ڵ�����

	//Shi-Tomasi�ǵ���

	for (int j = 0; j < rawNumOfWin; j++)
	{
		int yCoordinate = j * winHeight;
		for (int i = 0; i < colNumOfWin; i++)
		{
			int xCoordinate = i * winWidth;

			//��ԭͼ�����ΪN������
			Rect rect(xCoordinate, yCoordinate, winWidth, winHeight);
			Mat roi(mat, rect);

			//��ȡ�ǵ�
			goodFeaturesToTrack(roi, corner, maxCornerNumber, qualityLevel, minDistance, Mat(), blockSize, false, k);

			//��ÿһ������Ľǵ�����
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

//---------------------LK�������������ɵ�켣����---------------------

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

	//�ǵ�켣����ÿһ�д���һ���ǵ㣬����ɾ��׷��ʧ�ܵĽǵ�

	Mat Y(2 * range, (int)sP, CV_64F);
	int jj = 0;
	int ii = 0;
	int count = 0;
	for (size_t j = 0; j < sP; j++)
	{
		if (status[j] == 0)	//�ж��Ƿ�׷��ʧ�ܣ�ʧ������ӽ�����
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

	//���·���켣�����С

	Rect rect(0, 0, count, 2 * range);
	Mat temp(Y, rect);
	W.push_back(temp.clone());
}

//-----------------------�洢׷�ٳɹ��Ľǵ�--------------------------

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


