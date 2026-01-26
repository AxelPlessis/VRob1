
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

	cout << endl << "Chemin passe a OpenCV : [" << path << "]" << endl << endl;

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
	fs["cameraMatrix"] >> K;
	fs["dist_coeffs"] >> dist;
	fs.release();
	std::cout << "Matrice camera K:\n" << K << endl << endl;
	std::cout << "dist:\n" << dist << endl << endl;


	// POINTS EN PIXELS ET EN CENTIMETRE

	vector<Point2f> zero_x_i_ref = {
		Point2f(376, 96),
		Point2f(497, 72),
		Point2f(558, 266),
		Point2f(433, 299)
	};
	
	vector<Point3f> w_X_i_ref = {
		{0,0, 0},
		{2.3,0, 0},
		{2.3,2., 0},
		{0,2., 0}
	};

	/*vector<Point3f> objectPoints = {
		{0,0, 0},
		{2.9,0, 0},
		{2.0,2.3, 0},
		{0,2.8, 0}
	};*/

	vector<Point2f> refPts = transform(zero_x_i_ref, K);

	cout << "refpts:" << refPts << endl;


	//  CONSTRUCTION DE LA MATRICE HOMOGENE cam_T_w
	
	Mat rvec, tvec;
	solvePnP(w_X_i_ref, zero_x_i_ref, K, dist, rvec, tvec);
	//cout << "rvec: " << rvec.t() << endl;
	//cout << "tvec: " << tvec.t() << endl;

	Mat R;
	Rodrigues(rvec, R);
	cout << "R: " << R << endl << endl;

	// Repere plan
	cv::Mat zero_T_w = (cv::Mat_<double>(4, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), tvec.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), tvec.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), tvec.at<double>(2),
		0, 0, 0, 1
	);
	cout << "Matrice homogene de la camera:" << endl << zero_T_w << endl << endl;
	

	//  DEBUT DU SIFT

	Ptr<SIFT> sift = SIFT::create();

	std::vector<KeyPoint> keypoints;
	Mat descriptors;
	
	sift->detectAndCompute(frame, noArray(), keypoints, descriptors);

	Mat output;
	drawKeypoints(frame, keypoints, output, Scalar::all(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// ==== IMAGE AVEC POINTS SIFT ====
	//imshow("SIFT keypoints", output);



	//   RECUPERATION SEULEMENT POINTS SIFTS DANS LA ZONE 

	// Coins + padding (pour accueillir les kp importants)
	vector<Point> zone = {
		Point(366, 86),
		Point(507, 62),
		Point(568, 276),
		Point(423, 309)
	};

	vector<KeyPoint> filtered_kp;
	Mat desc_zero;

	for (int i = 0; i < keypoints.size(); i++) {
		Point2f p = keypoints[i].pt;

		double inside = pointPolygonTest(zone, p, false);

		if (inside >= 0) {  
			filtered_kp.push_back(keypoints[i]);
			desc_zero.push_back(descriptors.row(i));
		}
	}

	Mat imgColor; 
	cvtColor(frame, imgColor, COLOR_BGR2BGRA);
	vector<vector<Point>> contours = {zone};
	polylines(imgColor, contours, true, Scalar(0, 255, 0), 2);

	drawKeypoints(imgColor, filtered_kp, imgColor, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// ==== IMAGE AVEC KEYPOINTS DANS LA ZONE ====
	//imshow("KP in zone", imgColor);


	//  CALCUL DES POSITIONS DES POINTS EN 3D DANS REPERE Rw

	vector<Point3d> w_X_i;

	cv::Mat R_inv = R.t();
	cv::Mat t_inv = -R_inv * tvec;

	cv::Mat w_T_zero = cv::Mat::eye(4, 4, CV_64F);

	R_inv.copyTo(w_T_zero(cv::Rect(0, 0, 3, 3)));
	t_inv.copyTo(w_T_zero(cv::Rect(3, 0, 1, 3)));

	cv::Point3d O_cam(w_T_zero.at<double>(0, 3),
		w_T_zero.at<double>(1, 3),
		w_T_zero.at<double>(2, 3));

	for (int i = 0; i < filtered_kp.size(); i++) {
		double x_norm = (filtered_kp[i].pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
		double y_norm = (filtered_kp[i].pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1);

		cv::Mat ray_cam = (cv::Mat_<double>(3, 1) << x_norm, y_norm, 1.0);

		cv::Mat ray_world_mat = R_inv * ray_cam;
		cv::Point3d d_world(ray_world_mat.at<double>(0),
			ray_world_mat.at<double>(1),
			ray_world_mat.at<double>(2));

		if (std::abs(d_world.z) > 1e-6) {
			double lambda = -O_cam.z / d_world.z;
			cv::Point3d X_world = O_cam + lambda * d_world;
			w_X_i.push_back(X_world);
		}
	}

	// --- VÉRIFICATION PAR REPROJECTION ---

	vector<Point3f> w_X_i_float;
	for (const auto& p : w_X_i) {
		w_X_i_float.push_back(Point3f((float)p.x, (float)p.y, (float)p.z));
	}
	vector<Point2f> projectedPoints;

	try {
		if (!w_X_i_float.empty()) {
			// K, dist, rvec, tvec peuvent rester en double (CV_64F)
			cv::projectPoints(w_X_i_float, rvec, tvec, K, dist, projectedPoints);
		}
	}
	catch (const cv::Exception& e) {
		cerr << "Erreur : " << e.what() << endl;
	}

	Mat imgCheck = frame.clone();
	for (size_t i = 0; i < projectedPoints.size(); i++) {
		circle(imgCheck, filtered_kp[i].pt, 4, Scalar(0, 0, 255), -1);
		
		circle(imgCheck, projectedPoints[i], 2, Scalar(0, 255, 0), -1);
	}

	// ==== IMAGE DE REPROJECTION ====
	imshow("Verification Reprojection", imgCheck);

	if (!w_X_i_float.empty()) {
		cout << "Exemple de point 3D calcule (Rw) : " << w_X_i_float[0] << endl;
	}

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	if (desc_zero.type() != CV_32F) desc_zero.convertTo(desc_zero, CV_32F);

	//  LECTURE DE LA VIDEO
	

	while (true) {

		// Commenter = une seule frame
		cap >> frame; 

		// Repeter en boucle la video
		if (frame.empty()) {
			break;
		}

		// SIFT
		vector<KeyPoint> keypoint_k;
		Mat desc_k;
		sift->detectAndCompute(frame, noArray(), keypoint_k, desc_k);

		if (!desc_k.empty()) {

			vector<vector<DMatch>> knn_matches;
			matcher->knnMatch(desc_zero, desc_k, knn_matches, 1);

			vector<Point2f> k_x_i;
			vector<Point3f> k_X_i;

			for (size_t i = 0; i < knn_matches.size(); i++) {
				k_x_i.push_back(keypoint_k[i].pt);
				k_X_i.push_back(w_X_i_float[i]);
			}

			if (k_x_i.size() >= 4) {
				Mat rvec_k, tvec_k;

				solvePnP(k_X_i, k_x_i, K, dist, rvec_k, tvec_k);
				drawFrameAxes(frame, K, dist, rvec_k, tvec_k, 1.0);
			}
		}

		
		imshow("Video Frame", frame);

		if (waitKey(30) >= 0) {
			break;
		}
	}
	cap.release();
	//destroyAllWindows();
	return 0;
}