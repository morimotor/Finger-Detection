#include <opencv2/opencv.hpp>
#include "Labeling.h"
#include <math.h>
#include <vector>
#include <crtdbg.h>
#include <algorithm>

using namespace std;

// ���W�A���x�ϊ�
#define rad_to_deg(rad) rad * 180.0f / PI
#define deg_to_rad(deg) deg * PI / 180.0f

// �E�B���h�E�T�C�Y
#define WINDOW_SIZE_X 320
#define WINDOW_SIZE_Y 240

// ���x�����O�ŏ��T�C�Y
#define MIN_SIZE 500

// �����_�̊Ԋu
#define POINT_SPACING 14

// �w��̍ő�p�x
#define MIN_DEG 110

// �w�̕t�����̍ŏ��p�x
#define MAX_DEG 260



// HSV�̊e�v�f�̎w�肵��loewr�`upper�̐F�𒊏o
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

	//3Ch��LUT�쐬
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
		//LUT�̐ݒ�
		cvSet1D(lut, i, cvScalar(val[0], val[1], val[2]));
	}

	//3Ch���Ƃ�LUT�ϊ��i�e�`�����l�����ƂɂQ�l�������j
	cvLUT(hsv_img, hsv_img, lut);
	cvReleaseMat(&lut);

	//�e�`�����l�����Ƃ�IplImage
	ch1_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	ch2_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	ch3_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);

	//�`�����l�����Ƃɓ�l�����ꂽ�摜�����ꂼ��̃`�����l���ɕ���
	cvSplit(hsv_img, ch1_img, ch2_img, ch3_img, NULL);

	//3Ch�S�Ă�AND�����A�}�X�N�摜���쐬
	mask_img = cvCreateImage(cvGetSize(hsv_img), hsv_img->depth, 1);
	cvAnd(ch1_img, ch2_img, mask_img);
	cvAnd(mask_img, ch3_img, mask_img);

	//���͉摜�̃}�X�N�̈���o�͉摜�փR�s�[
	cvZero(dst_img);
	cvCopy(src_img, dst_img, mask_img);

	//���
	cvReleaseImage(&hsv_img);
	cvReleaseImage(&ch1_img);
	cvReleaseImage(&ch2_img);
	cvReleaseImage(&ch3_img);
	cvReleaseImage(&mask_img);

}

// 2�̃x�N�g�����炻�̂Ȃ��p���o��
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
	// �摜
	IplImage *input_img;
	IplImage *temp_img;
	IplImage *nichi_img;

	// HSV���o�͈́iH,S,V�j
	int lower[3] = {0, 50, 0};
	int upper[3] = {20, 255, 255};

	// ���x�����O
	LabelingBS label;
	LabelingBS::RegionInfo *regInfo;
	LabelingBS::RSPIterator rspIterator;


	// ���
	float gx, gy;	// �d�S
	int max_size = 0;

	input_img = cvLoadImage("file002.png", CV_LOAD_IMAGE_COLOR);
	temp_img = cvCreateImage(cvSize(WINDOW_SIZE_X, WINDOW_SIZE_Y), IPL_DEPTH_8U, 3);
	nichi_img = cvCreateImage(cvSize(WINDOW_SIZE_X, WINDOW_SIZE_Y), IPL_DEPTH_8U, 1);

	cvNamedWindow("���͉摜", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("�o�͉摜", CV_WINDOW_AUTOSIZE);




	HsvColorExtraction(input_img, temp_img, lower, upper);

	cvThreshold(temp_img, temp_img, 1, 255, CV_THRESH_BINARY);

	// �c���E���k�Ō�����
	cvDilate(temp_img, temp_img);
	//cvErode(temp_img, temp_img);

	cvSmooth(temp_img, temp_img, CV_MEDIAN);


	// 8bit��l�摜�쐬(���x�����O�̂���)
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

	// ���x�����O����
	short *buff = new short[WINDOW_SIZE_X * WINDOW_SIZE_Y];
	label.Exec((unsigned char *)(nichi_img->imageData), buff, WINDOW_SIZE_X, WINDOW_SIZE_Y, true, MIN_SIZE);

	gx = gy = -1000.0f;
	for (int i = 0; i<label.GetNumOfRegions(); i++)
	{
		float _gx, _gy;
		int _rw, _rh;
		int _max_x, _max_y, _min_x, _min_y;

		regInfo = label.GetResultRegionInfo(i);
		int nop = regInfo->GetNumOfPixels();		// �̈�̖ʐ�(��f��) 
		regInfo->GetCenterOfGravity(_gx, _gy);		// �̈�̏d�S
		regInfo->GetSize(_rw, _rh);					// �̈�̊O�ڋ�`�T�C�Y
		regInfo->GetMax(_max_x, _max_y);			// �̈�̉E�����W
		regInfo->GetMin(_min_x, _min_y);			// �̈�̍�����W


		if (max_size < nop)max_size = nop;	// MAX�T�C�Y�X�V

		// ���ň͂�
		if (i == 0)cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(0, 255, 255));
		else if (i == 1)cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(0, 255, 0));
		else cvRectangle(temp_img, cvPoint(_min_x, _min_y), cvPoint(_max_x, _max_y), cvScalar(255, 255, 0));

		if (i == 0)
		{
			gx = _gx;
			gy = _gy;
			printf("�d�S(%f, %f) �ʐ� : %d(pix) �T�C�Y(%d, %d)\n", _gx, _gy, nop, _rw, _rh);
		}

		if (gx < 0 || gx > WINDOW_SIZE_X)gx = -1.0f;
		if (gy < 0 || gy > WINDOW_SIZE_Y)gy = -1.0f;
	}

	// �d�S
	cvCircle(temp_img, cvPoint(gx, gy), 5, cvScalar(0, 255, 0), -1, CV_AA);

	// ���x�����O�̍ő�T�C�Y�̂��݂̂̂��c��
	for (int i = WINDOW_SIZE_X * WINDOW_SIZE_Y; i--;)
	{
		switch (buff[i])
		{
		case 1: nichi_img->imageData[i] = 255; break;
		default: nichi_img->imageData[i] = 0;
		}
	}

	// �֊s
	CvMemStorage *storage = cvCreateMemStorage(0); //�������X�g���[�W
	CvSeq *contours = 0; //�V�[�P���X
	CvTreeNodeIterator it;


	// �֊s����̓_
	int find_contour_num = cvFindContours(nichi_img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//cvDrawContours(temp_img, contours, CV_RGB(255, 1, 1), CV_RGB(255, 1, 1), 1, 2, CV_AA, cvPoint(0, 0));	// �֊s�`��
	
	if (max_size > MIN_SIZE)
	{

		// �c���[�m�[�h�C�e���[�^�̏�����
		cvInitTreeNodeIterator(&it, contours, 1);

		int numCounter = 0;	// �A�Ԃ̃J�E���^
		while ((contours = (CvSeq *)cvNextTreeNode(&it)) != NULL && contours->total > POINT_SPACING * 2)
		{
			// �֊s���\�����钸�_���W���擾
			CvPoint *prepre = CV_GET_SEQ_ELEM(CvPoint, contours, (POINT_SPACING)* (-2));	// 2�O�̓_
			CvPoint *pre = CV_GET_SEQ_ELEM(CvPoint, contours, -(POINT_SPACING));			// 1�O�̓_

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

				// �w��
				if (deg < MIN_DEG)
				{
					//cvEllipse(temp_img, *pre, cvSize(10, 10), end_angle, 0, -deg, CV_RGB(0, 255, 0), -1, 8, 0);
					cvCircle(temp_img, *pre, 6, cvScalar(255, 0, 0), -1, CV_AA);

				}
				// �w�̕t����
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
	







	// �摜�\��
	cvShowImage("���͉摜", input_img);
	cvShowImage("�o�͉摜", temp_img);

	// �C�ӂ̃L�[���͂�����ΏI��
	cvWaitKey(-1);
	

	// �㏈��
	cvReleaseImage(&input_img);
	cvReleaseImage(&temp_img);
	cvReleaseImage(&nichi_img);
	cvReleaseMemStorage(&storage);
	cvDestroyAllWindows();

	return 0;
}