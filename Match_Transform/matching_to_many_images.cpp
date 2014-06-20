#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"



#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const string defaultDetectorType = "SURF";
const string defaultDescriptorType = "SURF";
const string defaultMatcherType = "FlannBased";
const string defaultQueryImageName = "../../opencv/samples/cpp/matching_to_many_images/query.png";
const string defaultFileWithTrainImages = "../../opencv/samples/cpp/matching_to_many_images/train/trainImages.txt";
const string defaultDirToSaveResImages = "../../opencv/samples/cpp/matching_to_many_images/results";

static void printPrompt( const string& applName )
{
    cout << "/*\n"
         << " * This is a sample on matching descriptors detected on one image to descriptors detected in image set.\n"
         << " * So we have one query image and several train images. For each keypoint descriptor of query image\n"
         << " * the one nearest train descriptor is found the entire collection of train images. To visualize the result\n"
         << " * of matching we save images, each of which combines query and train image with matches between them (if they exist).\n"
         << " * Match is drawn as line between corresponding points. Count of all matches is equel to count of\n"
         << " * query keypoints, so we have the same count of lines in all set of result images (but not for each result\n"
         << " * (train) image).\n"
         << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [detectorType] [descriptorType] [matcherType] [queryImage] [fileWithTrainImages] [dirToSaveResImages]" << endl;
    cout << endl;

    cout << "\nExample:" << endl
         << "./" << applName << " " << defaultDetectorType << " " << defaultDescriptorType << " " << defaultMatcherType << " "
         << defaultQueryImageName << " " << defaultFileWithTrainImages << " " << defaultDirToSaveResImages << endl;
}

static void displayTransform( const Mat& queryImage, const Mat& trainImage, 
                        const vector<Point2f> obj, const vector<Point2f> scene, const string& resultDir, const string& name)
{

    Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( queryImage.cols, 0 );
    obj_corners[2] = cvPoint( queryImage.cols, queryImage.rows ); obj_corners[3] = cvPoint( 0, queryImage.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    Size sz1 = queryImage.size();
    Size sz2 = trainImage.size();
    Mat im3(sz2.height, sz1.width+sz2.width, CV_8UC3);
    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
    queryImage.copyTo(left);
    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
    trainImage.copyTo(right);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( im3, scene_corners[0] + Point2f( queryImage.cols, 0), scene_corners[1] + Point2f( queryImage.cols, 0), Scalar( 0, 255, 0), 7 );
    line( im3, scene_corners[1] + Point2f( queryImage.cols, 0), scene_corners[2] + Point2f( queryImage.cols, 0), Scalar( 0, 255, 0), 7 );
    line( im3, scene_corners[2] + Point2f( queryImage.cols, 0), scene_corners[3] + Point2f( queryImage.cols, 0), Scalar( 0, 255, 0), 7 );
    line( im3, scene_corners[3] + Point2f( queryImage.cols, 0), scene_corners[0] + Point2f( queryImage.cols, 0), Scalar( 0, 255, 0), 7 );

    string filename = resultDir + "transformed_" + name;
    imwrite(filename, im3);
}

struct kp {
    float dist;
    int index;
    Point2f matchingp;
    kp(){};
    kp(float distance, int ind, Point2f p){
        dist = distance;
        index = ind;
        matchingp = p;
    }
};

class compare_1 { // simple comparison function
   public:
      bool operator()(const Point2f& a,const Point2f& b) const { return (a.x- b.x)>0; } // returns x>y
};

static void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask,
                    vector<KeyPoint> queryKeypoints, vector<vector<KeyPoint> > trainKeypoints, const Mat& queryImage, 
                    const Mat& trainImage, const string& resultDir, const string& name )
{
    mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );

    vector<KeyPoint> trains = trainKeypoints.at(trainImgIdx);

    map<Point2f, kp, compare_1> trainmap;
    map<Point2f, kp, compare_1> querymap;

    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
        {
            Point2f trainp = trains[matches[i].trainIdx].pt;
            Point2f queryp = queryKeypoints[matches[i].queryIdx].pt;
            if (trainmap.count(trainp) != 0)
            {
                kp current(trainmap[trainp]);
                if (current.dist > matches[i].distance)
                {
                    mask[current.index] = 0;
                    trainmap[trainp] = kp(matches[i].distance, i, queryp);
                    mask[i] = 1;
                }
            }
            else 
            {
                trainmap[trainp] = kp(matches[i].distance, i, queryp);
                mask[i]= 1;
            }
            if (querymap.count(queryp) !=0)
            {
                kp c(querymap[queryp]);
                if (c.dist > matches[i].distance)
                {
                    mask[c.index] = 0;
                    querymap[queryp] = kp(matches[i].distance, i, trainp);
                    mask[i] = 1;
                }
            }
            else
            {
                querymap[queryp] = kp(matches[i].distance, i, trainp);
                mask[i]=1;
            }
        }
    }

    vector<Point2f> obj;
    vector<Point2f> scene;

    for (map<Point2f, kp>::iterator i = trainmap.begin(); i != trainmap.end(); ++i)
    {
        scene.push_back(i->first);
        obj.push_back((i->second).matchingp);
    }

    displayTransform(queryImage, trainImage, obj, scene, resultDir, name);
}

static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == string::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

static bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
                                      Ptr<FeatureDetector>& featureDetector,
                                      Ptr<DescriptorExtractor>& descriptorExtractor,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
    // Threshold nonmaxsuppresion type
    // featureDetector = new DynamicAdaptedFeatureDetector(cv::AdjusterAdapter::create("FAST"), 3500, 3550, 100);
    featureDetector = FeatureDetector::create( detectorType );
    descriptorExtractor = DescriptorExtractor::create( descriptorType );
    descriptorMatcher = DescriptorMatcher::create( matcherType );
    cout << ">" << endl;

    bool isCreated = featureDetector && descriptorExtractor && descriptorMatcher;
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

    return isCreated;
}

static bool readImages( const string& queryImageName, const string& trainFilename,
                 Mat& queryImage, vector <Mat>& trainImages, vector<string>& trainImageNames )
{
    cout << "< Reading the images..." << endl;
    queryImage = imread( queryImageName);//, IMREAD_GRAYSCALE);
    if( queryImage.empty() )
    {
        cout << "Query image can not be read." << endl << ">" << endl;
        return false;
    }
    string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        cout << "Train image filenames can not be read." << endl << ">" << endl;
        return false;
    }
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ )
    {
        string filename = trainDirName + trainImageNames[i];
        Mat img = imread( filename);//, IMREAD_GRAYSCALE );
        if( img.empty() )
            cout << "Train image " << filename << " can not be read." << endl;
        else
            readImageCount++;
        trainImages.push_back( img );
    }
    if( !readImageCount )
    {
        cout << "All train images can not be read." << endl << ">" << endl;
        return false;
    }
    else
        cout << readImageCount << " train images were read." << endl;
    cout << ">" << endl;

    return true;
}

static void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                      Ptr<FeatureDetector>& featureDetector )
{
    cout << endl << "< Extracting keypoints from images..." << endl;
    featureDetector->detect( queryImage, queryKeypoints );
    featureDetector->detect( trainImages, trainKeypoints );
    cout << ">" << endl;
}

static void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
                         const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                         Ptr<DescriptorExtractor>& descriptorExtractor )
{
    cout << "< Computing descriptors for keypoints..." << endl;
    descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
    descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

    int totalTrainDesc = 0;
    for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
        totalTrainDesc += tdIter->rows;

    cout << "Query descriptors count: " << queryDescriptors.rows << "; Total train descriptors count: " << totalTrainDesc << endl;
    cout << ">" << endl;
}

static void matchDescriptors( const Mat& queryDescriptors, const vector<Mat>& trainDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    TickMeter tm;

    tm.start();
    descriptorMatcher->add( trainDescriptors );
    descriptorMatcher->train();
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();
    descriptorMatcher->match( queryDescriptors, matches );
    tm.stop();
    double matchTime = tm.getTimeMilli();

    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

    cout << "Number of matches: " << matches.size() << endl;
    cout << "Build time: " << buildTime << " ms; Match time: " << matchTime << " ms" << endl;
    cout << ">" << endl;
}

static void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
                       const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir )
{
    cout << "< Save results..." << endl;
    Mat drawImg;
    vector<char> mask;
    for( size_t i = 0; i < trainImages.size(); i++ )
    {
        if( !trainImages[i].empty() )
        {
            maskMatchesByTrainImgIdx( matches, (int)i, mask, queryKeypoints, trainKeypoints, queryImage, trainImages[i], resultDir, trainImagesNames[i]);
            drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
                         matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask );
            string filename = resultDir + "/res_" + trainImagesNames[i];
            if( !imwrite( filename, drawImg ) )
                cout << "Image " << filename << " can not be saved (maybe because directory " << resultDir << " does not exist)." << endl;
        }
    }
    cout << ">" << endl;
}

int main(int argc, char** argv)
{
    string detectorType = defaultDetectorType;
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string queryImageName = defaultQueryImageName;
    string fileWithTrainImages = defaultFileWithTrainImages;
    string dirToSaveResImages = defaultDirToSaveResImages;

    if( argc != 7 && argc != 1 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    cv::initModule_nonfree();

    if( argc != 1 )
    {
        detectorType = argv[1]; descriptorType = argv[2]; matcherType = argv[3];
        queryImageName = argv[4]; fileWithTrainImages = argv[5];
        dirToSaveResImages = argv[6];
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
    vector<Mat> trainImages;
    vector<string> trainImagesNames;
    if( !readImages( queryImageName, fileWithTrainImages, queryImage, trainImages, trainImagesNames ) )
    {
        printPrompt( argv[0] );
        return -1;
    }

    vector<KeyPoint> queryKeypoints;
    vector<vector<KeyPoint> > trainKeypoints;
    detectKeypoints( queryImage, queryKeypoints, trainImages, trainKeypoints, featureDetector );

    Mat queryDescriptors;
    vector<Mat> trainDescriptors;
    computeDescriptors( queryImage, queryKeypoints, queryDescriptors,
                        trainImages, trainKeypoints, trainDescriptors,
                        descriptorExtractor );

    vector<DMatch> matches;
    matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );

    saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
                      matches, trainImagesNames, dirToSaveResImages );
    return 0;
}