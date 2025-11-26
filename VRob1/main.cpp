
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

	vector<Point2f> coins;

    if (!cap.isOpened()) {
		cerr << "Error: Could not open video file." << endl;
		return -1;
	}

	namedWindow("Video Frame");
	setMouseCallback("Video Frame", onMouse);

	Mat frame;

	cap >> frame;

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