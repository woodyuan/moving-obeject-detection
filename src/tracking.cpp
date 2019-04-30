#include "try.h"

using namespace std;
using namespace cv;

//----------------------------------ORB追踪---------------------------------

//计算给定点集横纵坐标极值
void computeCor(const vector<Point2f> &Point, float &maxX, float &maxY, float &minX, float &minY)
{
	size_t sP = Point.size();
	maxX = (Point[0].x > 0)? Point[0].x : 0;
	maxY = (Point[0].y > 0)? Point[0].y : 0;
	minX = (Point[0].x < 1000 && Point[0].x > 0) ? Point[0].x : 1000;
	minY = (Point[0].y < 1000 && Point[0].y > 0) ? Point[0].y : 1000;

	for (size_t i = 1; i < sP; i++)
	{
		if (Point[i].x > maxX)
			maxX = Point[i].x;
		if (Point[i].y > maxY)
			maxY = Point[i].y;
		if (Point[i].x < minX && Point[i].x > 0)
			minX = Point[i].x;
		if (Point[i].y < minY && Point[i].y > 0)
			minY = Point[i].y;
	}
}

//计算图像序列特征点和区域
void track(const vector<Mat> &srcGrayImage,vector<vector<Point2f> > &featurePoint, vector<vector<Point2f> > &orbPoint, vector<vector<float> > &featureCor, vector<vector<float> > &orbCor)
{
	float maxFX, maxFY, minFX, minFY;
	float maxOX, maxOY, minOX, minOY;
	float reMaxFX, reMinFX, reMaxFY, reMinFY;

	int maxCornerNumber = 80;		//区域内角点最大数量
	double qualityLevel = 0.01;		//角点检测可接受的最小特征值
	double minDistance = 10;		    //角点之间的最小距离
	int blockSize = 3;				//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;				//权重系数

	
	int maxLevel = 3;
	size_t sP = featurePoint[0].size();
	int Flag = 0;
	double minEigThreshold = 0.001;
	Size winSize(9, 9);
	TermCriteria Termcrit(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01);
	vector<Point2f> preFeature = featurePoint[0];
	vector<uchar> preStatus(sP);
	vector<uchar> status;
	vector<float> error;

	fill(preStatus.begin(), preStatus.end(), 1);
	computeCor(featurePoint[0], maxFX, maxFY, minFX, minFY);

	reMaxFX = maxFX;
	reMinFX = minFX;
	reMaxFY = maxFY;
	reMinFY = minFY;

	maxFX += 10;
	maxFY += 10;
	minFX -= 10;
	minFY -= 10;

	featureCor[0].push_back(maxFX);
	featureCor[0].push_back(minFX);
	featureCor[0].push_back(maxFY);
	featureCor[0].push_back(minFY);

	//-- 初始化
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	for (size_t i = 0; i < srcGrayImage.size() - 1; i++)
	{
		Mat img_1 = srcGrayImage[i].clone();
		Mat img_2 = srcGrayImage[i + 1].clone();

		//-- 第一步:检测 Oriented FAST 角点位置
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		//-- 第二步:根据角点位置计算 BRIEF 描述子
		descriptor->compute(img_1, keypoints_1, descriptors_1);
		descriptor->compute(img_2, keypoints_2, descriptors_2);

		//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
		vector<DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(descriptors_1, descriptors_2, matches);

		//-- 第四步:匹配点对筛选
		double min_dist = 10000, max_dist = 0;

		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
		for (int j = 0; j < descriptors_1.rows; j++)
		{
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
		Rect rec0(reMinFX, reMinFY,reMaxFX - reMinFX, reMaxFY - reMinFY);
		vector<Point2f> po(0);

		for (int j = 0; j < descriptors_1.rows; j++)
		{
			if (matches[j].distance <= max(2 * min_dist, 30.0))
			{
				int pl = matches[j].queryIdx;
				int sl = matches[j].trainIdx;
				Point2f temp0;
				Point2f temp1;
				temp0 = keypoints_1[pl].pt;
				if (rec0.contains(temp0))
				{
					temp0.x = (float)cvRound(keypoints_1[pl].pt.x);
					temp0.y = (float)cvRound(keypoints_1[pl].pt.y);
					temp1.x = (float)cvRound(keypoints_2[sl].pt.x);
					temp1.y = (float)cvRound(keypoints_2[sl].pt.y);

					orbPoint[i].push_back(temp0);
					po.push_back(temp1);
					if (i == srcGrayImage.size() - 2)
						orbPoint[i+1].push_back(temp1);
				}
			}
		}
		
		computeCor(orbPoint[i], maxOX, maxOY, minOX, minOY);
		maxOX += 10;
		maxOY += 10;
		minOX -= 10;
		minOY -= 10;

		orbCor[i].push_back(maxOX);
		orbCor[i].push_back(minOX);
		orbCor[i].push_back(maxOY);
		orbCor[i].push_back(minOY);

		computeCor(po, maxOX, maxOY, minOX, minOY);

		maxOX += 20;
		maxOY += 20;
	    minOX -= 20;
		minOY -= 20;

		vector<Point2f> feature;    
		vector<Point2f> track;
	    Rect rec1(minOX, minOY, maxOX - minOX, maxOY - minOY);

		featurePoint[i+1].resize(sP);
		fill(featurePoint[i+1].begin(), featurePoint[i+1].end(), Point2f(-100, -100));
		calcOpticalFlowPyrLK(srcGrayImage[i], srcGrayImage[i + 1], preFeature, track, status, error, winSize, maxLevel, Termcrit, Flag, minEigThreshold);
		fill(preFeature.begin(), preFeature.end(), Point2f(-100, -100));

		for (size_t j = 0; j < sP; j++)
		{
			if (preStatus[j] == 0)
				status[j] = 0;
			else if  (!(rec1.contains(track[j])))
				status[j] = 0;
		}

		for (size_t k = 0; k < sP; k++)
		{
			if (status[k] == 1)
			{
				Point2f temp0;
				temp0.x = cvRound(track[k].x);
				temp0.y = cvRound(track[k].y);
				featurePoint[i + 1][k] = temp0;
				preFeature[k] = track[k];
			}
		}

		computeCor(preFeature, reMaxFX, reMaxFY, reMinFX, reMinFY);
		
		Rect rec2(reMinFX , reMinFY, reMaxFX - reMinFX, reMaxFY - reMinFY);
		cornerDetection(srcGrayImage[i + 1], feature);

		for (size_t j = 0; j < feature.size(); j++)
		{
			if (rec2.contains(feature[j]))
			{
				bool fresh = true;
				for (size_t k = 0; k < sP; k++)
				{
					if (feature[j] == featurePoint[i + 1][k])
					{
						fresh = false;
						break;
					}
				}
				if (fresh)
				{
					featurePoint[i + 1].push_back(feature[j]);
					preFeature.push_back(feature[j]);
					status.push_back(1);
				}
			}
		}

		sP = featurePoint[i + 1].size();
		preStatus = status;

		computeCor(preFeature, reMaxFX, reMaxFY, reMinFX, reMinFY);

		maxFX = cvRound(reMaxFX + 10);
		maxFY = cvRound(reMaxFY + 10);
	    minFX = cvRound(reMinFX - 10);
		minFY = cvRound(reMinFY - 10);

		featureCor[i + 1].push_back(maxFX);
		featureCor[i + 1].push_back(minFX);
		featureCor[i + 1].push_back(maxFY);
		featureCor[i + 1].push_back(minFY);

		cout << i << endl;

		if (i == srcGrayImage.size() - 2)
		{
			computeCor(orbPoint[i+1], maxOX, maxOY, minOX, minOY);

			maxOX += 10;
			maxOY += 10;
			minOX -= 10;
			minOY -= 10;

			orbCor[i + 1].push_back(maxOX);
			orbCor[i + 1].push_back(minOX);
			orbCor[i + 1].push_back(maxOY);
			orbCor[i + 1].push_back(minOY);

		}
	}
}