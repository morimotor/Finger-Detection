#include <opencv2/opencv.hpp>
#include "Labeling.h"
#include <math.h>
#include <vector>
#include <crtdbg.h>
#include <algorithm>

using namespace std;

// ラジアン度変換
#define rad_to_deg(rad) rad * 180.0f / PI
#define deg_to_rad(deg) deg * PI / 180.0f

// ウィンドウサイズ
#define WINDOW_SIZE_X 320
#define WINDOW_SIZE_Y 240

// ラベリング最小サイズ
#define MIN_SIZE 500

// 特徴点の間隔
#define POINT_SPACING 14

// 指先の最大角度
#define MIN_DEG 110

// 指の付け根の最小角度
#define MAX_DEG 260



// HSVの各要素の指定したloewr～upperの色を抽出
void HsvColorExtraction(IplImage* src_img, IplImage* dst_img, int *_lower, int *_upper)
{
	IplImage *hsv_img;
	IplImage *ch1_img, *ch2_img, *ch3_img;
	IplImage *mask_img;

	int lower[3];
	int upper[3];
	int val[3];

	CvMat *lut;

	hsv_img = cvCreateImage(cvGetSize(src_img), src_img->depth, src_img->nChannels);
	cvCvtColor(src_img, hsv_img, CV_BGR2HSV);

	//3ChのLUT作成
	lut = cvCreateMat(256, 1, CV_8UC3);

	lower[0] = _lower[0];
	lower[1] = _lower[1];
	lower[2] = _lower[2];

	upper[0] = _upper[0];
	upper[1] = _upper[1];
	upper[2] = _upper[2];

	for (int i = 0; i < 256; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			if (lower[k] <= upper[k])
			{
				if ((lower[k] <= i) && (i <= upper[k]))
					val[k] = 255;
				else
					val[k] = 0;
			}
			else
			{
				if ((i <= upper[k]) || (lower[k] <= i))
					val[k] = 255;
				else
					val[k] = 0;
			}
		}
		//LUTの設定
		cvSet1D(lut, i, cvScalar(val[0], val[1], val[2]));
	}

	//3ChごとのLUT変換（各チャンネルごとに２値化処理）
	cvLUT(hsv_img, hsv_img, lut);
	cvReleaseMat(&lut);

	//各チャンネルごとのIplImage
	ch1_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	ch2_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	ch3_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);

	//チャンネルごとに二値化された画像をそれぞれのチャンネルに分解
	cvSplit(hsv_img, ch1_img, ch2_img, ch3_img, NULL);

	//3Ch全てのANDを取り、マスク画像を作成
	mask_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	cvAnd(ch1_img, ch2_img, mask_img);
	cvAnd(mask_img, ch3_img, mask_img);

	//入力画像のマスク領域を出力画像へコピー
	cvZero(dst_img);
	cvCopy(src_img, dst_img, mask_img);

	//解放
	cvReleaseImage(&hsv_img);
	cvReleaseImage(&ch1_img);
	cvReleaseImage(&ch2_img);
	cvReleaseImage(&ch3_img);
	cvReleaseImage(&mask_img);

}

// 2つのベクトルからそのなす角を出す
float calcAngle(CvPoint first, CvPoint center, CvPoint last)
{
	CvPoint p1 = first, p2 = center, p3 = last, v1, v2;

	v1.x = p1.x - p2.x;
	v1.y = p1.y - p2.y;
	v2.x = p3.x - p2.x;
	v2.y = p3.y - p2.y;

	float s = (float)(v1.x * v2.y - v1.y * v2.x);
	float t = (float)(v1.x * v2.x + v1.y * v2.y);

	float rad = atan2(s, t);

	if (rad < 0.0f)rad += 2.0f * (float)CV_PI;

	float deg = rad * 180.0f / (float)CV_PI;

	//if (s < 0.0) deg = 360.0f - deg;

	return deg;
}

int main(void)
{
	// 画像
	IplImage *input_img;
	IplImage *temp_img;
	IplImage *nichi_img;

	// HSV抽出範囲（H,S,V）
	int lower[3] = {0, 50, 0};
	int upper[3] = {20, 255, 255};

	// ラベリング
	LabelingBS label;
	LabelingBS::RegionInfo *regInfo;
	LabelingBS::RSPIterator rspIterator;


	// 情報
	float gx, gy;	// 重心
	int max_size = 0;

	input_img = cvLoadImage("file002.png", CV_LOAD_IMAGE_COLOR);
	temp_img = cvCreateImage(cvSize(WINDOW_SIZE_X, WINDOW_SIZE_Y), IPL_DEPTH_8U, 3);
	nichi_img = cvCreateImage(cvSize(WINDOW_SIZE_X, WINDOW_SIZE_Y), IPL_DEPTH_8U, 1);

	cvNamedWindow("入力画像", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("出力画像", CV_WINDOW_AUTOSIZE);




	HsvColorExtraction(input_img, temp_img, lower, upper);

	cvThreshold(temp_img, temp_img, 1, 255, CV_THRESH_BINARY);

	// 膨張・収縮で穴埋め
	cvDilate(temp_img, temp_img);
	//cvErode(temp_img, temp_img);

	cvSmooth(temp_img, temp_img, CV_MEDIAN);


	// 8bit二値画像作成(ラベリングのため)
	for (int y = 0; y < WINDOW_SIZE_Y; y++)
	{
		for (int x = 0; x < WINDOW_SIZE_X; x++)
		{
			uchar p[3];
			p[0] = temp_img->imageData[temp_img->widthStep * y + x * 3 + 0];    // B
			p[1] = temp_img->imageData[temp_img->widthStep * y + x * 3 + 1];    // G
			p[2] = temp_img->imageData[temp_img->widthStep * y + x * 3 + 2];    // R

			if (p[0] == 255)
				nichi_img->imageData[nichi_img->widthStep * y + x] = 255;
			else
				nichi_img->imageData[nichi_img->widthStep * y + x] = 0;
		}
	}

	// ラベリング処理
	short *buff = new short[WINDOW_SIZE_X * WINDOW_SIZE_Y];
	label.Exec((unsigned char *)(nichi_img->imageData), buff, WINDOW_SIZE_X, WINDOW_SIZE_Y, true, MIN_SIZE);

	gx = gy = -1000.0f;
	for (int i = 0; i<label.GetNumOfRegions(); i++)
	{
		float _gx, _gy;
		int _rw, _rh;
		int _max_x, _max_y, _min_x, _min_y;

		regInfo = label.GetResultRegionInfo(i);
		int nop = regInfo->GetNumOfPixels();		// 領域の面積(画素数) 
		regInfo->GetCenterOfGravity(_gx, _gy);		// 領域の重心
		regInfo->GetSize(_rw, _rh);					// 領域の外接矩形サイズ
		regInfo->GetMax(_max_x, _max_y);			// 領域の右下座標
		regInfo->GetMin(_min_x, _min_y);			// 領域の左上座標


		if (max_size < nop)max_size = nop;	// MAXサイズ更新

		// □で囲う
		if (i == 0)cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(0, 255, 255));
		else if (i == 1)cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(0, 255, 0));
		else cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(255, 255, 0));

		if (i == 0)
		{
			gx = _gx;
			gy = _gy;
			printf("重心(%f, %f) 面積 : %d(pix) サイズ(%d, %d)\n", _gx, _gy, nop, _rw, _rh);
		}

		if (gx < 0 || gx > WINDOW_SIZE_X)gx = -1.0f;
		if (gy < 0 || gy > WINDOW_SIZE_Y)gy = -1.0f;
	}

	// 重心
	cvCircle(temp_img, cvPoint(gx, gy), 5, cvScalar(0, 255, 0), -1, CV_AA);

	// ラベリングの最大サイズのもののみを残す
	for (int i = WINDOW_SIZE_X * WINDOW_SIZE_Y; i--;)
	{
		switch (buff[i])
		{
		case 1: nichi_img->imageData[i] = 255; break;
		default: nichi_img->imageData[i] = 0;
		}
	}

	// 輪郭
	CvMemStorage *storage = cvCreateMemStorage(0); //メモリストレージ
	CvSeq *contours = 0; //シーケンス
	CvTreeNodeIterator it;


	// 輪郭周りの点
	int find_contour_num = cvFindContours(nichi_img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//cvDrawContours(temp_img, contours, CV_RGB(255, 1, 1), CV_RGB(255, 1, 1), 1, 2, CV_AA, cvPoint(0, 0));	// 輪郭描画
	
	if (max_size > MIN_SIZE)
	{

		// ツリーノードイテレータの初期化
		cvInitTreeNodeIterator(&it, contours, 1);

		int numCounter = 0;	// 連番のカウンタ
		while ((contours = (CvSeq *)cvNextTreeNode(&it)) != NULL && contours->total > POINT_SPACING * 2)
		{
			// 輪郭を構成する頂点座標を取得
			CvPoint *prepre = CV_GET_SEQ_ELEM(CvPoint, contours, (POINT_SPACING)* (-2));	// 2つ前の点
			CvPoint *pre = CV_GET_SEQ_ELEM(CvPoint, contours, -(POINT_SPACING));			// 1つ前の点

			CvPoint *point;

			cvLine(temp_img, *prepre, *pre, cvScalar(0, 255, 0));



			for (int i = 0; i < contours->total / POINT_SPACING; i++)
			{
				int c = i * POINT_SPACING;

				point = CV_GET_SEQ_ELEM(CvPoint, contours, c);

				cvCircle(temp_img, cvPoint((int)point->x, (int)point->y), 2, cvScalar(0, 255, 0), -1, CV_AA);

				cvLine(temp_img, *pre, *point, cvScalar(0, 255, 0));

				float deg = calcAngle(*prepre, *pre, *point);
				float end_angle = calcAngle(cvPoint(pre->x + 1, pre->y), *pre, *point);

				// 指先
				if (deg < MIN_DEG)
				{
					//cvEllipse(temp_img, *pre, cvSize(10, 10), end_angle, 0, -deg, CV_RGB(0, 255, 0), -1, 8, 0);
					cvCircle(temp_img, *pre, 6, cvScalar(255, 0, 0), -1, CV_AA);

				}
				// 指の付け根
				else if (deg > MAX_DEG)
				{
					//cvEllipse(temp_img, *pre, cvSize(10, 10), end_angle, 0, -deg, CV_RGB(0, 0, 255), -1, 8, 0);
					cvCircle(temp_img, *pre, 6, cvScalar(100, 0, 100), -1, CV_AA);
				}

				numCounter++;
				prepre = pre;
				pre = point;
			}
		}
	}
	







	// 画像表示
	cvShowImage("入力画像", input_img);
	cvShowImage("出力画像", temp_img);

	// 任意のキー入力があれば終了
	cvWaitKey(-1);
	

	// 後処理
	cvReleaseImage(&input_img);
	cvReleaseImage(&temp_img);
	cvReleaseImage(&nichi_img);
	cvReleaseMemStorage(&storage);
	cvDestroyAllWindows();
	delete buff;
	return 0;
}
