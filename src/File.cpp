#include "File.h"

bool is_dir(const path& p) {
	return is_directory(p);
}

bool is_file(const path& p) {
	return is_regular_file(p);
}

std::vector<path> iterate(const std::string dir, std::function<bool(const path&)> is) {
	std::vector<path> list;
	path p(dir);
	recursive_directory_iterator iter(p), eod;
	BOOST_FOREACH(path const& i, std::make_pair(iter, eod)) {
		if (is(i)) {
			list.push_back(i);
		}
	}

	return list;
}

std::vector<path> getDirList(const std::string& dir) {
	return iterate(dir, std::bind(is_dir, _1));
}

std::vector<path> getFileList(const std::string& dir) {
	return iterate(dir, std::bind(is_file, _1));
}

void getAll(const path& root, const std::string& ext, std::vector<std::string>& ret) {
	if (!exists(root) || !is_directory(root)) return;

	recursive_directory_iterator it(root);
	recursive_directory_iterator endit;

	while (it != endit) {
		if (is_regular_file(*it))
			ret.push_back(it->path().string());
		++it;
	}
}

void write_vector_to_file(const std::vector<std::string>& v, std::string filename) {
	std::ofstream out(filename, std::ios::app);
	std::ostream_iterator<std::string> iter(out, "\n");
	std::copy(v.begin(), v.end(), iter);
}

bool saveMatrix(const std::string& filename, const Mat& matrix, const std::string& name) {
	FileStorage fs(filename, FileStorage::APPEND);
	if (fs.isOpened()) {
		fs << name << matrix;
		fs.release();
		return true;
	}
	return false;
}

bool readMatrix(const std::string& filename, Mat& matrix, const std::string& matrixname)
{
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs[matrixname] >> matrix;
		return !matrix.empty();
	}
	return false;
}

/*
Compute average value
*/
float average(std::vector<float> values)
{
	float sum = 0;
	for (float var : values)
	{
		sum += var;
	}
	return sum / values.size();
}

float getHistogramIntersection(Mat a, Mat b) {
	float result = 0;

#pragma omp parallel for shared(result)
	for (int i = 0; i < a.cols; i++) {
		result += min(a.at<float>(0, i), b.at<float>(0, i));
	}
	return result;
}

void getMedianHistogram(const std::vector<Mat> histograms, Mat& output) {
	getHistogram(histograms, output, average);
}

void getHistogram(const std::vector<Mat> histograms, Mat& output, float(*func)(std::vector<float> values)) {
	int count = histograms.size();
	if (count == 0) return;
	std::cout << count << std::endl;

	int columns = histograms[0].cols;
	float multiplier = 0;
	output = Mat(1, columns, CV_32F);

	//for each column create median
#pragma omp parallel for shared(output, multiplier)
	for (int i = 0; i < columns; i++)
	{
		std::vector<float> values;
		for (int j = 0; j < count; j++)
			values.push_back(histograms[j].at<float>(0, i));
		float value = func(values);
		multiplier += value;
		output.at<float>(0, i) = value;
	}

	// normalize histogram
#pragma omp parallel for
	for (int i = 0; i < columns; i++)
		output.at<float>(0, i) = output.at<float>(0, i) / multiplier;

	}



void printHistogram(Mat histogram) {
	std::cout << "{";
	// each row
	for (int i = 0; i < histogram.rows; i++)
	{
		std::cout << "[";
		// each column
		for (int j = 0; j < histogram.cols; j++)
		{
			// print value as float number
			std::cout << histogram.at<float>(i, j);
			/*if (j != histogram.cols - 1)
				std::cout << ",";*/
		}
		std::cout << "]";
	}
	std::cout << "}" << std::endl;
}
