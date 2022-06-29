/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! SampleYolo.cpp
//! This file contains the implementation of the YOYOv4 sample. It creates the network using
//! the YOLOV4 ONNX model.
//!

#include "SampleYolo.hpp"

#include <chrono>

const std::string SampleYolo::gSampleName = "TensorRT.sample_yolo";

int calculate_num_boxes(int input_h, int input_w)
{
    int num_anchors = 3;

    int h1 = input_h / 8;
    int h2 = input_h / 16;
    int h3 = input_h / 32;

    int w1 = input_w / 8;
    int w2 = input_w / 16;
    int w3 = input_w / 32;

    return num_anchors * (h1 * w1 + h2 * w2 + h3 * w3);
}


long long SampleYolo::now_in_milliseconds()
{
    return std::chrono::duration_cast <std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();  
}


SampleYolo::SampleYolo(const SampleYoloParams& params)
        : 
        mParams(params),
        mEngine(nullptr),
        mCocoResult(this->mParams.cocoTestResultFileName, std::ofstream::out),
        mImageIdx(0)
{
    char str[100];

    std::ifstream coco_names(this->mParams.cocoClassNamesFileName);
    while (coco_names.getline(str, 100))
    {
        std::string cls_name{ str };
        this->mClasses.push_back(cls_name.substr(0, cls_name.size()));
    }
    coco_names.close();
     
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the YOLO network by parsing the ONNX model and builds
//!          the engine that will be used to run YOLO (this->mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleYolo::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    if (this->fileExists(mParams.engingFileName))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.engingFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
        {
            infer->setDLACore(mParams.dlaCore);
        }
        this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        infer->destroy();

        sample::gLogInfo << "TRT Engine loaded from: " << mParams.engingFileName << std::endl;

        std::cout << "**Bindings**" << std::endl;
        for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
        {
            nvinfer1::Dims dim = mEngine->getBindingDimensions(i);

            std::cout << "name: " << mEngine->getBindingName(i) << std::endl;
            std::cout << "size: ";
            for (int j = 0; j < dim.nbDims; j++)
            {
                std::cout << dim.d[j];
                if (j < dim.nbDims - 1)
                    std::cout << "x";
            }
            std::cout << std::endl;
        }

        std::cout << "Num of bindings in engine: " << mEngine->getNbBindings() << std::endl;

        if (!this->mEngine)
        {
            return false;
        }
        else
        {
            this->mInputDims.nbDims = this->mParams.inputShape.size();
            this->mInputDims.d[0] = this->mParams.inputShape[0];
            this->mInputDims.d[1] = this->mParams.inputShape[1];
            this->mInputDims.d[2] = this->mParams.inputShape[2];
            this->mInputDims.d[3] = this->mParams.inputShape[3];

            return true;
        }
    }
    else
    {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return false;
        }

        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
        {
            return false;
        }

        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            return false;
        }

        assert(network->getNbInputs() == 1);
        this->mInputDims = network->getInput(0)->getDimensions();
        std::cout << this->mInputDims.nbDims << std::endl;
        assert(this->mInputDims.nbDims == 4);
    }

    return true;
}

//!
//! \brief Uses an onnx parser to create the YOLO Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the YOLO network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleYolo::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;

    sample::gLogInfo << "Parsing ONNX file: " << mParams.onnxFileName << std::endl;

    if (!parser->parseFromFile(mParams.onnxFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << mParams.onnxFileName << std::endl;
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);

    config->setMaxWorkspaceSize(4096_MiB);

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    // issue for int8 mode
    if (mParams.int8)
    {
        BatchStream calibrationStream(
            mParams.explicitBatchSize, mParams.nbCalBatches, mParams.calibrationBatches, mParams.dataDirs);
        calibrator.reset(
            new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

    // Enable DLA if mParams.dlaCore is true
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    sample::gLogInfo << "Building TensorRT engine: " << mParams.engingFileName << std::endl;

    this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    
    if (!this->mEngine)
    {
        return false;
    }

    if (mParams.engingFileName.size() > 0)
    {
        std::ofstream p(mParams.engingFileName, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = this->mEngine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.engingFileName << std::endl;
    }
     
    return true;
}

bool SampleYolo::infer_iteration(SampleUniquePtr<nvinfer1::IExecutionContext> &context, samplesCommon::BufferManager &buffers, cv::Mat &image, cv::Mat &outputImage, int count)
{
    std::vector<BoundingBox> nms_bboxes;

    auto time1 = this->now_in_milliseconds();
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);

    if (!processInput_aspectRatio(buffers, image))
    {
        return false;
    }

    auto time2 = this->now_in_milliseconds();

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());

    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    auto time3 = this->now_in_milliseconds();

    // Post-process detections and verify results
    if (!verifyOutput_aspectRatio(buffers, image, nms_bboxes))
    {
        return false;
    }

    auto time4 = this->now_in_milliseconds();

    // Draw bboxes only for the first image in each batch

    this->draw_bboxes(nms_bboxes, outputImage);

    this->mSpeedInfo.preProcess += time2 - time1;
    this->mSpeedInfo.model += time3 - time2;
    this->mSpeedInfo.postProcess += time4 - time3;

    // Calculate fps
    auto totalRuntime = this->mSpeedInfo.preProcess + this->mSpeedInfo.model + this->mSpeedInfo.postProcess;
    float fps = count / (totalRuntime / 1000.0);

    cv::putText(outputImage, std::to_string((int)fps) + " fps", cv::Point(20, 50), 1.2, 2, cv::Scalar(0, 255, 255), 2);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleYolo::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(this->mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    cv::VideoCapture cap(this->mParams.inputVideoName);

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;

    while (cap.read(frame))
    {
        cv::Mat output_frame = frame.clone();

        if (this->infer_iteration(context, buffers, frame, output_frame, ++frameCount))
        {
            cv::imshow("Result", output_frame);
            cv::waitKey(1);
        }
        else
        {
            cap.release();
            return false;
        }
    }
    cap.release();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleYolo::processInput_aspectRatio(const samplesCommon::BufferManager& buffers, cv::Mat &mSampleImage)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(this->mParams.inputTensorNames[0]));

    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    this->image_rows.clear();
    this->image_cols.clear();
    this->image_pad_rows.clear();
    this->image_pad_cols.clear();

    cv::Mat rgb_img;

    // Convert BGR to RGB
    cv::cvtColor(mSampleImage, rgb_img, cv::COLOR_BGR2RGB);

    auto scaleSize = cv::Size(inputW, inputH);
    cv::Mat resized;
    cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::split(resized, input_channels[b]);
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }

    return true;
}


//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleYolo::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(this->mParams.inputTensorNames[0]));

    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    this->image_rows.clear();
    this->image_cols.clear();
    this->image_pad_rows.clear();
    this->image_pad_cols.clear();

    cv::Mat rgb_img;

    // Convert BGR to RGB
    cv::cvtColor(this->mSampleImage, rgb_img, cv::COLOR_BGR2RGB);

    auto scaleSize = cv::Size(inputW, inputH);
    cv::Mat resized;
    cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::split(resized, input_channels[b]);
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleYolo::verifyOutput_aspectRatio(const samplesCommon::BufferManager& buffers, cv::Mat &image, std::vector<BoundingBox> & nms_bboxes)
{
    const int keepTopK = mParams.keepTopK;

    float *output = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[0]));

    if (!output)
    {
        std::cout << "NULL value output detected!" << std::endl;
        return false;
    }

    nms_bboxes = this->get_bboxes(this->mParams.outputShapes[0][0], keepTopK, output);

    return true;
}


std::vector<BoundingBox> SampleYolo::get_bboxes(int batch_size, int keep_topk, float *output)
{
    std::vector<BoundingBox> bboxes;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> rectBoxes;

    int i = 0;
    int nc = 80;
    while (i < 8400)
    {
        // Box
        int k = i * 85;
        float object_conf = output[k + 4];

        if (object_conf < this->mParams.confThreshold)
        {
            i++;
            continue;
        }

        // (center x, center y, width, height) to (x, y, w, h)
        float x = (output[k] - output[k + 2] / 2);
        float y = (output[k + 1] - output[k + 3] / 2);
        float width = output[k + 2];
        float height = output[k + 3];

        // Classes
        float class_conf = output[k + 5];
        int classId = 0;

        for (int j = 1; j < nc; j++)
        {
            if (class_conf < output[k + 5 + j])
            {
                classId = j;
                class_conf = output[k + 5 + j];
            }
        }
        
        i++;
        
        class_conf *= object_conf;

        classIds.push_back(classId);
        confidences.push_back(class_conf);
        rectBoxes.emplace_back(cv::Rect((int)x, (int)y, (int)width, (int)height));

    }

    // Non-maximum suppression to eliminate redudant overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rectBoxes, confidences, this->mParams.confThreshold, this->mParams.nmsThreshold, indices);
    bboxes.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        BoundingBox box;
        // (x, y, w, h) to (x1, y1, x2, y2)
        box.x = rectBoxes[indices[i]].x;
        box.y = rectBoxes[indices[i]].y;
        box.w = rectBoxes[indices[i]].width;
        box.h = rectBoxes[indices[i]].height;
        box.score = confidences[indices[i]];
        box.cls = classIds[indices[i]];

        bboxes.emplace_back(box);
    }

    return bboxes;
}

void SampleYolo::draw_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg)
{
    int H = testImg.rows;
    int W = testImg.cols;

    for (size_t k = 0; k < bboxes.size(); k++)
    {
        if (bboxes[k].cls == -1)
        {
            break;
        }

        int x = (bboxes[k].x / 640) * W;
        int y = (bboxes[k].y / 640) * H;
        int w = (bboxes[k].w / 640) * W;
        int h = (bboxes[k].h / 640) * H;

        auto box_rect = cv::Rect(x, y, w, h);
        auto color = colors[bboxes[k].cls % colors.size()];
        
        cv::rectangle(testImg, box_rect, color, 2);
        //cv::putText(testImg, this->mClasses[bboxes[k].cls], cv::Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.8, color, 1);
        cv::putText(testImg, this->mClasses[bboxes[k].cls], cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

    }
}


