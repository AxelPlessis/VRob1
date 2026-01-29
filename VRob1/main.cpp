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

// Donne coordonnées en pixel quand on clique sur un point sur video frame
void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Point2f(" << x << ", " << y << ")" << endl;
	}
}

// Transforme des points image en coordonnées normalisées caméra
vector<Point2f> transform(vector<Point2f> imgpts, Mat cameraMatrix)
{
	vector<Point2f> refPts(imgpts.size());
	for (int i = 0; i < imgpts.size(); i++) {
		refPts[i].x = (imgpts[i].x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
		refPts[i].y = (imgpts[i].y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(1, 1);
	}
	return refPts;
}

// Charge la calibration caméra depuis le fichier XML (Etape 3)
bool loadCalibration(const string& path, Mat& K, Mat& dist)
{
	FileStorage fs(path, FileStorage::READ);
	if (!fs.isOpened()) return false;
	fs["cameraMatrix"] >> K;
	fs["dist_coeffs"] >> dist;
	fs.release();
	return true;
}

// Calcul de la matrice homogene 0_T_w avec Rodrigue et solvePnP (Etape 5)
void computePose(
	const vector<Point3f>& w_X_i_ref,
	const vector<Point2f>& zero_x_i_ref,
	const Mat& K,
	const Mat& dist,
	Mat& rvec,
	Mat& tvec,
	Mat& R,
	Mat& zero_T_w
)
{
	// Estimation rotation + translation
	solvePnP(w_X_i_ref, zero_x_i_ref, K, dist, rvec, tvec);
	
	// Passage du vecteur rvec de rotation à une matrice R
	Rodrigues(rvec, R);

	// Construction de la matrice homogene
	zero_T_w = (Mat_<double>(4, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), tvec.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), tvec.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), tvec.at<double>(2),
		0, 0, 0, 1
		);
}

// SIFT (Etape 6)
void computeSIFT(
	const Mat& frame,
	Ptr<SIFT>& sift,
	vector<KeyPoint>& keypoints,
	Mat& descriptors
)
{
	sift->detectAndCompute(frame, noArray(), keypoints, descriptors);
}

// Recupere les points clés uniquement dans la zone d'intérêt (Etape 6)
void filterKeypointsInZone(
	const vector<KeyPoint>& keypoints,
	const Mat& descriptors,
	const vector<Point2f>& zero_x_i_ref,
	vector<KeyPoint>& zero_x_i,
	Mat& desc_zero
)
{
	for (int i = 0; i < keypoints.size(); i++) {
		Point2f p = keypoints[i].pt;
		double inside = pointPolygonTest(zero_x_i_ref, p, false);
		if (inside >= 0) {
			zero_x_i.push_back(keypoints[i]);
			desc_zero.push_back(descriptors.row(i));
		}
	}
}

// Affiche la zone d'intérêt et les keypoints conservés (Etape 6)
void drawZoneAndKeypoints(
	const Mat& frame,
	const vector<Point2f>& zero_x_i_ref,
	const vector<KeyPoint>& zero_x_i
)
{
	Mat imgColor;
	cvtColor(frame, imgColor, COLOR_BGR2BGRA);

	vector<Point> contours(zero_x_i_ref.begin(), zero_x_i_ref.end());
	polylines(imgColor, vector<vector<Point>>{contours}, true, Scalar(0, 255, 0), 2);
	drawKeypoints(imgColor, zero_x_i, imgColor, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("KP in zone", imgColor);
}

// Reconstruction des points 3D avec plan Z = 0 établi (Etape 7)
vector<Point3d> compute3DPoints(
	const vector<KeyPoint>& zero_x_i,
	const Mat& K,
	const Mat& R,
	const Mat& tvec
)
{
	vector<Point3d> w_X_i;

	// Inversion de la pose caméra
	Mat R_inv = R.t();
	Mat t_inv = -R_inv * tvec;

	// Matrice homogène w_T_0
	Mat w_T_zero = Mat::eye(4, 4, CV_64F);
	R_inv.copyTo(w_T_zero(Rect(0, 0, 3, 3)));
	t_inv.copyTo(w_T_zero(Rect(3, 0, 1, 3)));

	// Position de la caméra dans le monde
	Point3d O_cam(
		w_T_zero.at<double>(0, 3),
		w_T_zero.at<double>(1, 3),
		w_T_zero.at<double>(2, 3)
	);

	for (int i = 0; i < zero_x_i.size(); i++) {
		// Coordonnées normalisées
		double x_norm = (zero_x_i[i].pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
		double y_norm = (zero_x_i[i].pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1);

		// 0_r_i
		Mat ray_cam = (Mat_<double>(3, 1) << x_norm, y_norm, 1.0);
		// w_r_i
		Mat ray_world_mat = R_inv * ray_cam;

		Point3d d_world(
			ray_world_mat.at<double>(0),
			ray_world_mat.at<double>(1),
			ray_world_mat.at<double>(2)
		);

		// Intersection avec le plan Z=0
		if (abs(d_world.z) > 1e-6) {
			double lambda = -O_cam.z / d_world.z;
			w_X_i.push_back(O_cam + lambda * d_world);
		}
	}

	return w_X_i;
}

// Reprojection pour vérifier que les points 3D se superposent aux points 2D (Etape 7)
// Rouge = point détecté 2D  | Vert = reprojection 3D
void reprojectAndDisplay(
	const Mat& frame,
	const vector<Point3d>& w_X_i,
	const vector<KeyPoint>& zero_x_i,
	const Mat& rvec,
	const Mat& tvec,
	const Mat& K,
	const Mat& dist
)
{
	vector<Point3f> w_X_i_float;
	for (const auto& p : w_X_i)
		w_X_i_float.emplace_back((float)p.x, (float)p.y, (float)p.z);

	vector<Point2f> projectedPoints;
	if (!w_X_i_float.empty())
		projectPoints(w_X_i_float, rvec, tvec, K, dist, projectedPoints);

	Mat imgCheck = frame.clone();
	for (size_t i = 0; i < projectedPoints.size(); i++) {
		circle(imgCheck, zero_x_i[i].pt, 4, Scalar(0, 0, 255), -1);
		circle(imgCheck, projectedPoints[i], 2, Scalar(0, 255, 0), -1);
	}

	imshow("Verification Reprojection", imgCheck);
}

// Writer de video pour l'export
VideoWriter initVideoWriter(const string& filename, int width, int height, double fps = 30.0)
{
	int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v'); // mp4
	VideoWriter writer(filename, fourcc, fps, Size(width, height));
	if (!writer.isOpened()) {
		cerr << "Erreur : impossible d'ouvrir le fichier vidéo pour écriture : " << filename << endl;
	}
	return writer;
}

int main()
{
	auto be = videoio_registry::getBackends();
	for (auto b : be) cout << videoio_registry::getBackendName(b) << "\n";

	int choice;
	cout << "Choisir la video (1 ou 2) : ";
	cin >> choice;

	string path;
	vector<Point2f> zero_x_i_ref;
	vector<Point3f> w_X_i_ref;

	switch (choice)
	{
	case 1:
		path = "./ressources/video1.mp4";
		zero_x_i_ref = {
			Point2f(335, 88),
			Point2f(492, 86),
			Point2f(499, 317),
			Point2f(339, 317)
		};
		w_X_i_ref = {
			{0,0,0},{12.5,0,0},{12.5,17.8,0},{0,17.8,0}
		};
		break;

	case 2:
		path = "./ressources/video2.mp4";
		zero_x_i_ref = {
			Point2f(324,119),
			Point2f(582,99),
			Point2f(597,353),
			Point2f(347,370)
		};
		w_X_i_ref = {
			{0,0,0},{25.5,0,0},{25.5,25.5,0},{0,25.5,0}
		};
		break;

	default:
		return -1;
	}

	cout << endl << "Chemin passe a OpenCV : [" << path << "]" << endl << endl;

	VideoCapture cap(path, CAP_FFMPEG);
	if (!cap.isOpened()) return -1;

	int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(CAP_PROP_FPS);

	VideoWriter writer = initVideoWriter("./output/result.mp4", width, height, fps);

	namedWindow("Video Frame");
	setMouseCallback("Video Frame", onMouse);

	// First frame
	Mat frame;
	cap >> frame;

	// Calibration (Etape 3)
	Mat K, dist;
	if (!loadCalibration("./ressources/calib.xml", K, dist)) return -1;

	// (Etape 4)
	vector<Point2f> refPts = transform(zero_x_i_ref, K);
	cout << "refpts:" << refPts << endl;

	// Calcul matrice homogene (Etape 5)
	Mat rvec, tvec, R, zero_T_w;
	computePose(w_X_i_ref, zero_x_i_ref, K, dist, rvec, tvec, R, zero_T_w);

	cout << "R: " << R << endl << endl;
	cout << "Matrice homogene de la camera:" << endl << zero_T_w << endl << endl;

	// SIFT (Etape 6)
	Ptr<SIFT> sift = SIFT::create();

	vector<KeyPoint> keypoints;
	Mat descriptors;
	computeSIFT(frame, sift, keypoints, descriptors);

	Mat output;
	// ---- Affichage des points SIFT ----
	drawKeypoints(frame, keypoints, output, Scalar::all(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT keypoints", output);

	vector<KeyPoint> zero_x_i;
	Mat desc_zero;
	filterKeypointsInZone(keypoints, descriptors, zero_x_i_ref, zero_x_i, desc_zero);

	// ---- Affichage des points SIFT dans la zone d'intérêt ----
	drawZoneAndKeypoints(frame, zero_x_i_ref, zero_x_i);

	// Calcul points 3D (Etape 7)
	vector<Point3d> w_X_i = compute3DPoints(zero_x_i, K, R, tvec);
	reprojectAndDisplay(frame, w_X_i, zero_x_i, rvec, tvec, K, dist);


	// BOUCLE DE LECTURE DES FRAMES I_k (Etape 8)
	while (true)
	{
		cap >> frame;
		if (frame.empty()) break;

		vector<KeyPoint> k_x_i;
		Mat desc_k;
		computeSIFT(frame, sift, k_x_i, desc_k);

		FlannBasedMatcher matcher;
		vector<vector<DMatch>> desc_matches;
		matcher.knnMatch(desc_zero, desc_k, desc_matches, 2);

		// Matching entre les points (Etape 9)
		vector<Point2f> zero_goodMatches_i, k_goodMatches_i;
		for (size_t i = 0; i < desc_matches.size(); i++) {
			if (desc_matches[i][0].distance < 0.75 * desc_matches[i][1].distance) {
				zero_goodMatches_i.push_back(zero_x_i[desc_matches[i][0].queryIdx].pt);
				k_goodMatches_i.push_back(k_x_i[desc_matches[i][0].trainIdx].pt);
			}
		}

		// Affichage du repère et du quadrilatère (Etape 10, 11 et 12)
		if (zero_goodMatches_i.size() >= 4) {
			Mat H = findHomography(zero_goodMatches_i, k_goodMatches_i, RANSAC);
			if (!H.empty()) {
				vector<Point2f> projected_corners;
				perspectiveTransform(zero_x_i_ref, projected_corners, H);

				vector<Point> points_int(projected_corners.begin(), projected_corners.end());
				polylines(frame, vector<vector<Point>>{points_int}, true, Scalar(255, 0, 255), 2);

				Mat rvec_k, tvec_k, R_k, k_T_w;
				computePose(w_X_i_ref, projected_corners, K, dist, rvec_k, tvec_k, R_k, k_T_w);

				drawFrameAxes(frame, K, dist, rvec_k, tvec_k, 5.0);
			}
		}

		// ---- Affichage de la frame I_k avec quadrilatere et repere ----
		imshow("Video Frame", frame);

		// Encodage de la video (Etape 13)
		if (writer.isOpened()) writer.write(frame);

		if (waitKey(30) >= 0) break;
	}
	cap.release();
	if (writer.isOpened()) writer.release();
	destroyAllWindows();
	return 0;
};
