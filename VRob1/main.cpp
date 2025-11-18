
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/registry.hpp>
#include <fstream>
#include <vector>

int main()

{
	auto be = cv::videoio_registry::getBackends();
	for (auto b : be) std::cout << cv::videoio_registry::getBackendName(b) << "\n";

	std::string path = "./ressources/video2.mp4";
	std::ifstream test(path);
	if (!test.good()) {
		std::cerr << "File not found or inaccessible: " << path << std::endl;
		return -1;
	}

	std::cout << "Chemin passé à OpenCV : [" << path << "]" << std::endl;

	cv::VideoCapture cap(path, cv::CAP_FFMPEG);



    if (!cap.isOpened()) {
		std::cerr << "Error: Could not open video file." << std::endl;
		return -1;
	}

	cv::Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			break; // End of video
		}
		cv::imshow("Video Frame", frame);
		if (cv::waitKey(30) >= 0) {
			break; // Exit on any key press
		}
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}