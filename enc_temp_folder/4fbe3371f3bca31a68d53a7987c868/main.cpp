
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/registry.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Point clicked : (" << x << ", " << y << ")" << endl;
	}
}

vector<Point2f> transform(vector<Point2f> imgpts, Mat cameraMatrix)
{
	vector<Point2f> refPts(imgpts.size());
	for (int i = 0; i < imgpts.size(); i++) {
		refPts[i].x = (imgpts[i].x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
		refPts[i].y = (imgpts[i].y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(1, 1);
	}
	return refPts;
}

int main()

{
	auto be = videoio_registry::getBackends();
	for (auto b : be) cout << videoio_registry::getBackendName(b) << "\n";

	string path = "./ressources/video1.mp4";
	ifstream test(path);
	if (!test.good()) {
		cerr << "File not found or inaccessible: " << path << endl;
		return -1;
	}

	cout << "Chemin passé à OpenCV : [" << path << "]" << endl;

	VideoCapture cap(path, CAP_FFMPEG);

    if (!cap.isOpened()) {
		cerr << "Error: Could not open video file." << endl;
		return -1;
	}

	namedWindow("Video Frame");
	setMouseCallback("Video Frame", onMouse);

	Mat frame;

	cap >> frame;

	// Chargement du fichier XML
	cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
	cv::Mat cameraMatrix, rvec, tvec;
	fs["camera_matrix"] >> cameraMatrix;
	fs.release();

	vector<Point2f> corners = {
		Point2f(462, 105),
		Point2f(558, 106),
		Point2f(558, 220),
		Point2f(462, 220)
	};

	vector<Point2f> objectPoints = {
	{0,0},
	{2.0,0},
	{2.0,2.3},
	{0,2.3}
	};

	cout << "Camera Matrix: " << cameraMatrix << endl;

	cout << transform(corners, cameraMatrix) << endl;

	


	while (true) {

		// Commenter : une seule frame
		// cap >> frame; 

		if (frame.empty()) {
			break; // End of video
		}
		imshow("Video Frame", frame);
		if (waitKey(30) >= 0) {
			break; // Exit on any key press
		}
	}
	cap.release();
	destroyAllWindows();
	return 0;
}