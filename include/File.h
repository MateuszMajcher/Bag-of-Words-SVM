#pragma once
#ifndef FILE_H
#define FILE_H
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace boost::filesystem;
using namespace cv;

using namespace std::placeholders;

template<typename T>
void printVector(const T& t) {
	std::copy(t.cbegin(), t.cend(), std::ostream_iterator<typename T::value_type>(std::cout, "\n"));
}

std::vector<path> iterate(const std::string dir, std::function<bool(const path&)> is);

std::vector<path> getDirList(const std::string& dir);

std::vector<path> getFileList(const std::string& dir);

void getAll(const path& root, const std::string& ext, std::vector<std::string>& ret);

bool saveMatrix(const std::string& filename, const Mat& matrix, const std::string& name);

bool readMatrix(const std::string& filename, Mat& matrix, const std::string& name);

float getHistogramIntersection(Mat a, Mat b);

void getHistogram(const std::vector<Mat> histograms, Mat& output, float(*func)(std::vector<float> values));

void getMedianHistogram(const std::vector<Mat> histograms, Mat& output);

void printHistogram(Mat histogram);
#endif // !FILE_H
