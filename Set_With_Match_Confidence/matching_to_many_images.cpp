#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
//#include <features2d.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const string defaultDetectorType = "SURF";
const string defaultDescriptorType = "SURF";
const string defaultMatcherType = "FlannBased";
const string defaultQueryImageName = "../../opencv/samples/cpp/matching_to_many_images/query.png";
const string defaultFileWithsearchImage = "../../opencv/samples/cpp/matching_to_many_images/train/searchImage.txt";
const string defaultDirToSaveResImage = "../../opencv/samples/cpp/matching_to_many_images/results";

static void printPrompt( const string& applName )
{
    cout << "/*\n"
         << " * Matches images with intelligent matcher."
         << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [detectorType] [descriptorType] [matcherType] [queryImage] [trainImage] [dirToSaveResImage]" << endl;
    cout << endl;

    cout << "\nExample:" << endl
         << "./" << applName << " " << defaultDetectorType << " " << defaultDescriptorType << " " << defaultMatcherType << " "
         << defaultQueryImageName << " " << defaultFileWithsearchImage << " " << defaultDirToSaveResImage
         << endl;
}

// static void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, vector<char>& mask )
// {
//     mask.resize( matches.size() );
//     fill( mask.begin(), mask.end(), 0 );
//     for( size_t i = 0; i < matches.size(); i++ )
//     {
//         if( matches.imgIdx == trainImgIdx )
//             mask = 1;
//     }
// }

static bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
                                      Ptr<FeatureDetector>& featureDetector,
                                      Ptr<DescriptorExtractor>& descriptorExtractor,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
    // Threshold nonmaxsuppresion type
    featureDetector = new DynamicAdaptedFeatureDetector(cv::AdjusterAdapter::create("FAST"), 3500, 3550, 100);
    // featureDetector = FeatureDetector::create( detectorType );
    descriptorExtractor = DescriptorExtractor::create( descriptorType );
    descriptorMatcher = DescriptorMatcher::create( matcherType );
    cout << ">" << endl;

    bool isCreated = featureDetector && descriptorExtractor && descriptorMatcher;
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

    return isCreated;
}

static bool readImages( const string& queryImageName, const string& searchImageName,
                 Mat& queryImage, Mat& searchImage )
{
    cout << "< Reading the images..." << endl;
    queryImage = imread( queryImageName);//, IMREAD_GRAYSCALE);
    if( queryImage.empty() )
    {
        cout << "Query image can not be read." << endl << ">" << endl;
        return false;
    }
    searchImage = imread( searchImageName );//, IMREAD_GRAYSCALE);
    if( searchImage.empty() )
    {
        cout << "Search image can not be read." << endl << ">" << endl;
        return false;
    }

    return true;
}

static void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      const Mat& searchImage, vector<KeyPoint>& searchKeypoints,
                      Ptr<FeatureDetector>& featureDetector )
{
    cout << endl << "< Extracting keypoints from images..." << endl;
    featureDetector->detect( queryImage, queryKeypoints );
    featureDetector->detect( searchImage, searchKeypoints );
    cout << ">" << endl;
}

static void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
                         const Mat& searchImage, vector<KeyPoint>& searchKeypoints, Mat& searchDescriptors,
                         Ptr<DescriptorExtractor>& descriptorExtractor )
{
    cout << "< Computing descriptors for keypoints..." << endl;
    descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
    descriptorExtractor->compute( searchImage, searchKeypoints, searchDescriptors );

    cout << "Query descriptors count: " << queryDescriptors.rows << "; Total train descriptors count: " << searchDescriptors.rows << endl;
    cout << ">" << endl;
}

static void matchDescriptors( const Mat& queryDescriptors, const Mat& searchDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    descriptorMatcher->add( searchDescriptors );
    descriptorMatcher->train();
    waitKey();

    descriptorMatcher->match( queryDescriptors, matches );

    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

    cout << "Number of matches: " << matches.size() << endl;
    cout << ">" << endl;
}

static void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const Mat& searchImage, const vector<KeyPoint>& searchKeypoints,
                       const vector<DMatch>& matches, const string& searchImageName, const string& resultDir )
{
    cout << "< Save results..." << endl;
    Mat drawImg;
    vector<char> mask;
    if( !searchImage.empty() )
    {
   //     maskMatchesByTrainImgIdx( matches, mask );
        drawMatches( queryImage, queryKeypoints, searchImage, searchKeypoints,
                     matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask );
        string filename = resultDir + "/res_" + searchImageName;
        if( !imwrite( filename, drawImg ) )
            cout << "Image " << filename << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;
    }
    cout << ">" << endl;
}

int main(int argc, char** argv)
{
    string detectorType = defaultDetectorType;
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string queryImageName = defaultQueryImageName;
    string searchImageName = defaultFileWithsearchImage;
    string dirToSaveResImage = defaultDirToSaveResImage
;

    if( argc != 7 && argc != 1 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    cv::initModule_nonfree();

    if( argc != 1 )
    {
        detectorType = argv[1]; descriptorType = argv[2]; matcherType = argv[3];
        queryImageName = argv[4]; searchImageName = argv[5];
        dirToSaveResImage = argv[6];
    }

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    Ptr<DescriptorMatcher> descriptorMatcher;
    if( !createDetectorDescriptorMatcher( detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher ) )
    {
        printPrompt( argv[0] );
        return -1;
    }

    Mat queryImage;
    Mat searchImage;
    if( !readImages( queryImageName, searchImageName, queryImage, searchImage ) )
    {
        printPrompt( argv[0] );
        return -1;
    }

    vector<KeyPoint> queryKeypoints;
    vector<KeyPoint> searchKeypoints;
    detectKeypoints( queryImage, queryKeypoints, searchImage, searchKeypoints, featureDetector );

    Mat queryDescriptors;
    Mat searchDescriptors;
    computeDescriptors( queryImage, queryKeypoints, queryDescriptors,
                        searchImage, searchKeypoints, searchDescriptors,
                        descriptorExtractor );

    vector<DMatch> matches;
    matchDescriptors( queryDescriptors, searchDescriptors, matches, descriptorMatcher );

    saveResultImages( queryImage, queryKeypoints, searchImage, searchKeypoints,
                      matches, searchImageName, dirToSaveResImage );
    return 0;
}