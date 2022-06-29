#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string> 
#include <map>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "SampleYolo.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

static vector<string> classNames;
static vector<Scalar> colors;

#pragma execution_character_set( "utf-8" )

void printHelpInfo()
{
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--demo          This app will run demo if this option is set"
        << std::endl;
    std::cout << "--speed         This app will run speed test if this option is set"
        << std::endl;
    std::cout << "--coco          This app will run COCO dataset if this option is set"
        << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
}

SampleYoloParams specifyInputAndOutputNamesAndShapes(SampleYoloParams& params)
{
	params.inputShape = std::vector<int>{ params.explicitBatchSize, 3, 640, 640 };

	// Output shapes when BatchedNMSPlugin is available
	params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, 1});
	params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK, 4});
	params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});
	params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});

	// Output tensors when BatchedNMSPlugin is available
	params.outputTensorNames.push_back("outputs");

	return params;
}

SampleYoloParams initializeSampleParams(std::vector<std::string> args)
{
    SampleYoloParams params;

    // This argument is for calibration of int8
    // Int8 calibration is not available until now
    // You have to prepare samples for int8 calibration by yourself 
    params.nbCalBatches = 80;

    // The engine file to generate or to load
    // The engine file does not exist:
    //     This program will try to load onnx file and convert onnx into engine
    // The engine file exists:
    //     This program will load the engine file directly
    params.engingFileName = "yolov6t_fp32.engine";

    // The onnx file to load
    params.onnxFileName = "yolov6t.onnx";

    // Input tensor name of ONNX file & engine file
    params.inputTensorNames.push_back("image_arrays");

    // Old batch configuration, it is zero if explicitBatch flag is true for the tensorrt engine
    // May be deprecated in the future
    params.batchSize = 0;

    // topK parameter of BatchedNMSPlugin
    params.topK = 2000;
    
    // Threshold values
    params.confThreshold = 0.3;
    params.nmsThreshold = 0.5;

    // keepTopK parameter of BatchedNMSPlugin
    params.keepTopK = 1000;

    // Batch size, you can modify to other batch size values if needed
    params.explicitBatchSize = 1;

    // params.inputImageName = "E:/Downloads/demo.jpg";
    params.inputVideoName = "road_traffic.mp4";
    params.cocoClassNamesFileName = "coco.names";

    // Config number of DLA cores, -1 if there is no DLA core
    params.dlaCore = -1;

    for (auto& arg : args)
    {
        params.demo = 1;
        params.outputImageName = "demo_out.jpg";

        if (arg == "--int8")
        {
            params.int8 = true;
        }
        else if (arg == "--fp16")
        {
            params.fp16 = true;
        }
    }
    specifyInputAndOutputNamesAndShapes(params);

    return params;
}
int main(int argc, char** argv)
{
    std::vector<std::string> args;

    auto sampleTest = sample::gLogger.defineTest(SampleYolo::gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleYolo sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Yolo" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogInfo << "Loading or building yolo model done" << std::endl;

    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

	return 0;
}
