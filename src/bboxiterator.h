/**
 * @file bboxiterator.h
 *
 * Interface definitions for the bounding-box iterator.
 */
#ifndef BBOXITERATOR_H
#define BBOXITERATOR_H

#include <mlearn.h>
#include <opencv2/core/core.hpp>

class BBoxIterator : public ML::DataIterator {
private:
   std::vector<ML::DataEntry> _entries;
	std::vector<ML::DataLabel> _labels;

   int _channels;
   cv::Size _size;
   std::vector<cv::Mat> _faces;

public:
   BBoxIterator(const cv::Mat& image, const std::vector<cv::Rect>& rects, cv::Size size);
	~BBoxIterator() {};

   int num_samples() const { return _entries.size(); }
	int sample_size() const { return _channels * _size.width * _size.height; }
   const std::vector<ML::DataEntry>& entries() const { return _entries; }
	const std::vector<ML::DataLabel>& labels() const { return _labels; }

	void sample(ML::Matrix& X, int i);
};

#endif
