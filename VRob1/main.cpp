
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

	string path = "./ressources/video2.mp4";
	ifstream test(path);
	if (!test.good()) {
		cerr << "File not found or inaccessible: " << path << endl;
		return -1;
	}

	cout << "Chemin passe a OpenCV : [" << path << "]" << endl;

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
	cv::FileStorage fs("./ressources/calib.xml", cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Failed to open calib.xml\n";
		return -1;
	}

	cv::Mat K, dist;

	// Les noms doivent correspondre exactement au XML :
	fs["cameraMatrix"] >> K;        // OK
	fs["dist_coeffs"] >> dist;      // OK

	fs.release();

	std::cout << "K:\n" << K << "\n\n";
	std::cout << "dist:\n" << dist << "\n";

	vector<Point2f> corners = {
		Point2f(376, 96),
		Point2f(497, 72),
		Point2f(558, 266),
		Point2f(433, 299)
	};
	
	vector<Point3f> objectPoints = {
		{0,0, 0},
		{2.9,0, 0},
		{2.0,2.3, 0},
		{0,2.8, 0}
	};

	vector<Point2f> refPts = transform(corners, K);

	cout << "refpts:" << refPts << endl;

	//Construction de la matrice homogène
	Mat rvec, tvec;
	solvePnP(objectPoints, corners, K, dist, rvec, tvec);
	cout << "rvec: " << rvec.t() << endl;
	cout << "tvec: " << tvec.t() << endl;

	Mat R;
	Rodrigues(rvec, R);
	cout << "R: " << R << endl;

	cv::Mat H = (cv::Mat_<double>(4, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), tvec.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), tvec.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), tvec.at<double>(2),
		0, 0, 0, 1
	);
	cout << "Matrice homogene: \n" << H << endl;
	

	Ptr<SIFT> sift = SIFT::create();

	std::vector<KeyPoint> keypoints;
	Mat descriptors;

	sift->detectAndCompute(frame, noArray(), keypoints, descriptors);

	Mat output;
	drawKeypoints(frame, keypoints, output, Scalar::all(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT keypoints", output);

	sift->detectAndCompute(frame, noArray(), keypoints, descriptors);

	//Coins mais agrandis pour accueillir les kp importants
	vector<Point> zone = {
		Point(366, 86),
		Point(507, 62),
		Point(568, 276),
		Point(423, 309)
	};

	vector<KeyPoint> filteredKP;
	Mat filteredDesc;

	for (int i = 0; i < keypoints.size(); i++) {
		Point2f p = keypoints[i].pt;

		double inside = pointPolygonTest(zone, p, false);

		if (inside >= 0) {  
			filteredKP.push_back(keypoints[i]);
			filteredDesc.push_back(descriptors.row(i));
		}
	}

	Mat imgColor; 
	cvtColor(frame, imgColor, COLOR_BGR2BGRA);
	vector<vector<Point>> contours = { zone };
	polylines(imgColor, contours, true, Scalar(0, 255, 0), 2);

	drawKeypoints(imgColor, filteredKP, imgColor, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("KP in zone", imgColor);

	vector<Point3d> kp3d;

	//Calcul des points 3D X des kp importants par rapport au repere Rw
	for (int i = 0; i < filteredKP.size(); i++) {
		Mat X = (cv::Mat_<double>(4, 1) << filteredKP[i].pt.x, filteredKP[i].pt.y, 0.0, 1.0);

		Mat Xw = H * X;

		kp3d.push_back(cv::Point3d(Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2)));
	}

	while (true) {

		// Commenter : une seule frame
		//cap >> frame; 

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