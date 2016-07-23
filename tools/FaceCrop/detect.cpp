#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame, std::string outPath);
std::vector<std::string> open(std::string path);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "./haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;

// Function main
int main(int argc, char **argv)
{
    std::string dir;
    std::vector<std::string> files;
    uint32_t i;

    // Load the cascade
    if (!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading\n");
        return (-1);
    }

    if (argc != 3)
    {
      fprintf(stderr, "\nUsage: ./detect ./path/to/images/directory ./destination/path\n\n");
      return -1;
    }
    else
    {
      dir = argv[1];
    }

    files = open(dir);

    mkdir(argv[2], 0700);

    for (i = 0; i < files.size(); i++)
    {
      if (files[i] == "." || files[i] == "..") continue;

      // Read the image file
      Mat frame = imread(dir + "/" + files[i]);

      // Apply the classifier to the frame
      if (!frame.empty()){
          detectAndDisplay(frame, argv[2]);
      }
      else{
          printf(" --(!) No captured frame -- Break!\n\n");
          //return 1;
      }

    }

    return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame, std::string outPath)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.01, 4, 0, Size(30, 30));

    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
        crop = frame(roi_c);
        resize(crop, res, Size(256, 256)); // This will be needed later while saving images
        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        // Form a filename
        filename = "";
        stringstream ssfn;
        ssfn << outPath << "/" << filenumber << "_cropped.png";
        filename = ssfn.str();
        filenumber++;

        imwrite(filename, gray);

        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);

        //imshow("original", frame);

        if (!crop.empty())
        {
            //imshow("detected", crop);
            //waitKey(50);
        }
        else
            destroyWindow("detected");
    }

}


// save each file within a path to a vector for proecessing
std::vector<std::string> open(std::string path)
{
  DIR * dir;
  dirent * pDir;
  std::vector<std::string> files;

  dir = opendir(path.c_str());

  while ((pDir = readdir(dir)) != NULL)
  {
    files.push_back(pDir->d_name);
  }

  return files;
}
