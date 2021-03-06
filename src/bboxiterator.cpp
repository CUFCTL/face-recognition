/**
 * @file bboxiterator.cpp
 *
 * Implementation of the bounding-box iterator.
 */
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include "bboxiterator.h"



/**
 * Construct a bounding-box iterator from an image
 * and a list of bounding boxes.
 *
 * @param image
 * @param rects
 * @param size
 */
BBoxIterator::BBoxIterator(const cv::Mat& image, const std::vector<cv::Rect>& rects, cv::Size size)
{
   _channels = image.channels();
   _size = size;

   for ( auto& rect : rects ) {
      // append image
      cv::Mat face = image(rect);
      cv::resize(face, face, size);

      _faces.push_back(face);

      // append empty entry
      _entries.push_back(mlearn::DataEntry { "", "" });
   }
}



void BBoxIterator::sample(mlearn::Matrix& X, int i)
{
   assert(X.rows() == this->sample_size());

   for ( int j = 0; j < X.rows(); j++ ) {
      X.elem(j, i) = _faces[i].at<cv::Vec3b>(
         (j / _channels) / _size.width,
         (j / _channels) % _size.width
      )[_channels - 1 - (j % _channels)];
   }
}
