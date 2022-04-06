#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <filesystem>
#include <set>

using namespace std;
using namespace cv;
using namespace std::filesystem;

bool autoResize = false;
bool debug = false;
bool comparison = false;
bool showResult = false;
set<string> validOutputExtensions = {".jpg", ".jpeg", ".png"};

class MaskedMat {
public:
    Mat mat;
    Mat mask;
};
MaskedMat resize(const MaskedMat & images);
void progressBar(int i, int start, int finish);

int main(int argc, char** argv) {
    int seamWidth = -1;
    string maskPath;
    int opt;
    path outputPath;
    string usage = "Usage: " + string(argv[0]) + " -s seam_size [-m mask_image] [-h] [-a] [-d] [-c] [-p] [-o output_image] input_image";
    while ((opt = getopt(argc, argv, "s:m:hadcpo:")) != -1) {
        switch (opt) {
            case 's':
            try {
                int s = atoi(optarg); 
                if (s < 0) {
                    cout << "Seam size must be greater than 0" << endl;
                    exit(EXIT_FAILURE);
                }
                seamWidth = s;
                break;
            } catch(const std::exception& e) {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }
            case 'a':
                autoResize = true; break;
            case 'd':
                debug = true; break;
            case 'm':
                maskPath = optarg; break;
            case 'c':
                comparison = true; break;
            case 'p':
                showResult = true; break;
            case 'o':
                outputPath = path(optarg);
                if (outputPath.is_relative()) outputPath = absolute(outputPath);
                if (validOutputExtensions.find(outputPath.extension()) != validOutputExtensions.end()) {
                    cerr << "Invalid output path. The given extension was not one of [.jpg, .jpeg, .png]" << endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                cout << usage << endl << endl
                    << "  -s\t\t\tSpecifies the amount of horizontal pixels to remove. Must be smaller than the horizontal size of the input image.\n\t\t\t  Ignored if -a is set" << endl
                    << "  -m\t\t\tThe path to the image to use as mask (best effort, not guarantee that elements will be removed/preserved completely).\n\t\t\t  It has to be a grayscale image, with 0 for element to remove, 1 for elements to preserve and 0.5 for everything else" << endl
                    << "  -h\t\t\tDisplay this help message" << endl
                    << "  -a\t\t\tEnable auto seam size detection from mask. If no mask is provided the default is 10% of the input image width" << endl
                    << "  -d\t\t\tEnable debug features. Waits for user input after each operation" << endl
                    << "  -c\t\t\tShow a comparison between seam resize and regular linear resize. Only works if -p is selected" << endl
                    << "  -p\t\t\tShow seam resize result in multiple windows" << endl
                    << "  -o\t\t\tImage(path) where to save the final result. Supported extensions are [.jpg, .jpeg, .png]" << endl
                    << "  input_image\t\tThe path to the image that needs to be carved" << endl;
                exit(EXIT_SUCCESS); 
                break;
            default: /* '?' */
                cerr << usage << endl;
                exit(EXIT_FAILURE);
        }
    }

    if (optind >= argc) {
        cerr << "Expected image path after options" << endl;
        exit(EXIT_FAILURE);
    }
    Mat original = imread(argv[optind++], IMREAD_COLOR); // Reading input image
    if (original.empty()) {
        cerr << "Image could not be opened, missing or incorrect path: " << argv[optind-1] << endl;
        return 1;
    }
    if (seamWidth < 0) seamWidth = (int)(original.cols * 0.1); // Setting default seam_size if none was provided
    if (seamWidth >= original.cols) {
        cout << "Seam size must be smaller than input image width" << endl;
        exit(EXIT_FAILURE);
    }
    Mat cropped = original.clone();


    if (!debug && !showResult && outputPath.empty()) {
        cout << "No output/display flag selected [-o, -p, -d], the result will be discarded." << endl
            << "Continue anyway? [y/n]" << endl;
        string action;
        cin >> action;
        transform(action.begin(), action.end(), action.begin(), ::tolower);
        if (action != "y" && action != "yes" && action != "yup") {
            cout << "Aborting. Use -h for help" << endl;
            exit(EXIT_SUCCESS);
        }
    }

    Mat mask;
    if (maskPath != "?") {
        mask = imread(maskPath, IMREAD_GRAYSCALE);
    }
    if (mask.empty()) {
        cerr << "No mask file provided or wrong path, proceeding with empty mask" << endl;
        mask = Mat_<char>(original.size());
        mask += 1;
        autoResize = false;
    }
    cv::resize(mask, mask, original.size(), 0, 0, INTER_LINEAR); // Resizing mask to fit over input image
    double maxVal;
    minMaxLoc(mask, NULL, &maxVal, NULL, NULL);
    mask /= maxVal / 2; // Scaling mask such that 255 -> 2, 128 -> 1, 0 -> 0. This is used later to bias the weight calculation

    if (autoResize) { // If autosize search for the largest part of the mask and use that as the amount of pixels to remove
        seamWidth = 0;
        for (int r = 0; r < mask.rows; r++) {
            auto row = mask.ptr<char>(r);
            int acc = 0;
            for (int c = 0; c < mask.cols; c++)
                acc += (row[c] == 0);
            if (acc > seamWidth) seamWidth = acc;
        }
    }
    cout << "Image will be resized by: " << seamWidth << " px" << endl;

    // Scaling mask to bias weight calculation appropriately
    mask.convertTo(mask, CV_64F);
    mask *= 2;
    mask -= 1;

    // Iterate over the input image and the mas remove a 1px seam each time
    MaskedMat iterator = {cropped, mask};
    for (int i = 0; i < seamWidth; i++) {
        iterator = resize(iterator);
        progressBar(i, 0, seamWidth);
    }
    cropped = iterator.mat;
    cout << "Successfully resized image: " << original.size() << "  --->   " << cropped.size() << endl;

    if (!outputPath.empty()) {
        imwrite(outputPath, cropped);
        cout << "Result saved to: " << outputPath << endl;
    }

    if (showResult) {
        imshow("Cropped", cropped);
        if (comparison) {
            Mat shrinked;
            cv::resize(original, shrinked, Size(original.cols - 100, original.rows), 0, 0, INTER_LINEAR);
            imshow("Shrinked", shrinked);
        }
        imshow("Original", original);
        waitKey(0);
    }

    exit(EXIT_SUCCESS);
}

// Carve a 1px seam in the input image with weihts biased by mask
MaskedMat resize(const MaskedMat & images) {
    Mat original = images.mat;
    Mat mask = images.mask;
    Mat grayscale;
    cvtColor(original, grayscale, COLOR_RGB2GRAY);

    Mat VKernel = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat HKernel = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // Convolute over the input image to find both vertical and horizontal edges
    Mat verticalEdges;
    Mat horizontalEdges;
    filter2D(grayscale, verticalEdges, grayscale.depth(), VKernel);
    filter2D(grayscale, horizontalEdges, grayscale.depth(), HKernel);
    Mat edges = (verticalEdges + horizontalEdges) / (2.0); // Combine edges found
    edges.convertTo(edges, CV_64F);
    edges /= 255; // Scale edges appropriately
    edges = edges.mul(mask); // Bias edges with mask
    if (debug) imshow("Edges", edges);

    // Compute weights from edges from bottom to top (seam will move in opposite direction for removal)
    // To improve memory usage the Mat edges is reused for the weights
    double max = 0;
    for (int r = edges.rows-2; r >= 0; r--) {
        auto currRow = edges.ptr<double>(r);
        auto nextRow = edges.ptr<double>(r+1);
        for (int c = 0; c < edges.cols; c++) {
            auto record = nextRow[c];
            if (c < edges.cols-1 && nextRow[c+1] < record)
                record = nextRow[c+1];
            if (c > 0 && nextRow[c-1] < record)
                record = nextRow[c-1];
            
            currRow[c] += record;
            if (currRow[c] > max) max = currRow[c];
        }
    }
    edges /= max; // Normalize weights

    // Remove the pixels that belong to the path with smallest weight from top to bottom
    Mat test;
    if (debug) test = original.clone();
    int col;
    auto row = edges.ptr<double>(0);
    for (int i = 0; i < edges.cols; i++) if (row[i] < row[col]) col = i;
    for (int r = 0; r < edges.rows; r++) {
        if (debug) test.at<Vec3b>(r, col) = Vec3b(0, 255, 0);
        if (r+1 < edges.rows) {
            row = edges.ptr<double>(r+1);
            if (col > 0 && row[col-1] < row[col]) col -= 1;
            if (col < edges.cols-1 && row[col+1] < row[col]) col += 1;
        }
        if (col < edges.cols-1) {
            mask(Rect(col+1, r, edges.cols-col-1, 1)).copyTo(mask(Rect(col, r, edges.cols-col-1, 1)));
            original(Rect(col+1, r, edges.cols-col-1, 1)).copyTo(original(Rect(col, r, edges.cols-col-1, 1)));
        }
        edges.at<double>(r, col) = 1;
    }

    if (debug) {
        imshow("Test", test);
        double min;
        minMaxLoc(edges, &min, &max, NULL, NULL);
        edges -= min;
        edges /= (max - min);
        imshow("Weights", edges);
        Mat maskCopy = mask.clone();
        minMaxLoc(maskCopy, &min, &max, NULL, NULL);
        maskCopy -= min;
        maskCopy /= (max - min);
        imshow("Mask", maskCopy);
        waitKey(0);
    }

    auto newCols = original.cols-1;
    return {original(Rect(0, 0, newCols, original.rows)), mask(Rect(0, 0, newCols, original.rows))};
}

// Display a small progress bar
void progressBar(int i, int start, int finish) {
    if (i != start) cout << "\e[1A\e[K";
    switch (time(NULL) % 4) {
        case 0: cout << "\u25e2 "; break;
        case 1: cout << "\u25e3 "; break;
        case 2: cout << "\u25e4 "; break;
        case 3: cout << "\u25e5 "; break;
    }
    int numSymbols = 3;
    int barLength = 50;
    double percent = (i+1.0)/(finish-start);
    cout << "|";
    for (int j = 0; j < barLength; j++) {
        int pos = j*100.0/barLength;
        if (pos < int(percent*100)) cout << "\u2588";
        else if (pos == int(percent*100)) 
            switch (i % numSymbols) {
                case 0: cout << "\u2581"; break;
                case 1: cout << "\u2582"; break;
                case 2: cout << "\u2583"; break;
                case 3: cout << "\u2584"; break;
                case 4: cout << "\u2585"; break;
                case 5: cout << "\u2586"; break;
                case 6: cout << "\u2587"; break;
                default: cout << "?";
            }
        else cout << " ";
    }
    cout << "| " << 100*percent << " %" << endl;
}