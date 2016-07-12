#include "opencv2/objdetect.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
int filenumber;
string filename;

/** @function main */
int main( void )
{
    //VideoCapture capture;
    Mat frame = imread("./media/tswift.jpg");

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    // possibly make this for images
    //-- 2. Read the video stream
    //capture.open( "73.mp4" );
    //if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    //while ( capture.read(frame) )
    //{
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            //break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        int c = waitKey(10);
        if( (char)c == 27 ) { return 1; } // escape
    //}
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    string sstm;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0;
    int ac = 0;

    size_t ib = 0;
    int ab = 0;

    for ( ic = 0; ic < faces.size(); ic++ )
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = faces[ic].width;
        roi_c.height = faces[ic].height;

        ac = roi_c.width * roi_c.height;

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = faces[ib].width;
        roi_b.height = faces[ib].height;

        ab = roi_b.width * roi_b.height;

        if (ac > ab)
        {
          ib = ic;
          roi_b.x = faces[ib].x;
          roi_b.y = faces[ib].y;
          roi_b.width = faces[ib].width;
          roi_b.height = faces[ib].height;
        }

        crop = frame(roi_b);
        resize(crop, res, Size(28, 128), 0, 0, INTER_LINEAR);
        cvtColor(crop, gray, CV_BGR2GRAY);

        filename = "";
        stringstream ssfn;
        ssfn << filenumber << ".png";
        filename = ssfn.str();
        filenumber++;

        imwrite("./cropped" + filename, gray);

        Point pt1(faces[ic].x, faces[ic].y);
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }
    //-- Show what you got
    //sstm << "Crop Size Area: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    imshow("original", frame);

    if (!crop.empty())
    {
      imshow("detected", crop);
    }
    else
    {
      destroyWindow("detected");
    }

}
