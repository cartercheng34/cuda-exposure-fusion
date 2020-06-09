//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include<string>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/cuda_types.hpp"
#include "opencv2/core/cuda.inl.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


using namespace std;
using namespace cv;

const bool cpu = false;				////set false to turn off print for x and residual



vector<double> _para = { 1, 1, 1, 8 };

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here
void readImages(vector<Mat>& images, int scene_No)
{
    if (scene_No == 0) {
        int numImages = 7;
        static const char* filenames[] =
        {
          "ntu_lib/IMG_0323.jpg",
          "ntu_lib/IMG_0317.jpg",
          "ntu_lib/IMG_0318.jpg",
          "ntu_lib/IMG_0319.jpg",
          "ntu_lib/IMG_0320.jpg",
          "ntu_lib/IMG_0321.jpg",
          "ntu_lib/IMG_0322.jpg",
        };

        for (int i = 0; i < numImages; i++)
        {
            Mat im = imread(filenames[i]);
            images.push_back(im);
        }
    }
    else if (scene_No == 1) {
        int numImages = 16;
        static const char* filenames[] =
        {
          "scene2/SAM_0015.jpg",
          "scene2/SAM_0016.jpg",
          "scene2/SAM_0017.jpg",
          "scene2/SAM_0018.jpg",
          "scene2/SAM_0019.jpg",
          "scene2/SAM_0020.jpg",
          "scene2/SAM_0021.jpg",
          "scene2/SAM_0022.jpg",
          "scene2/SAM_0023.jpg",
          "scene2/SAM_0024.jpg",
          "scene2/SAM_0025.jpg",
          "scene2/SAM_0026.jpg",
          "scene2/SAM_0027.jpg",
          "scene2/SAM_0028.jpg",
          "scene2/SAM_0029.jpg",
          "scene2/SAM_0030.jpg",
        };

        for (int i = 0; i < numImages; i++)
        {
            Mat im = imread(filenames[i]);
            images.push_back(im);
        }
    }
    else if (scene_No == 2) {
        int numImages = 14;
        static const char* filenames[] =
        {
          "library/DSC00716.jpg",
          "library/DSC00717.jpg",
          "library/DSC00718.jpg",
          "library/DSC00719.jpg",
          "library/DSC00720.jpg",
          "library/DSC00721.jpg",
          "library/DSC00722.jpg",
          "library/DSC00723.jpg",
          "library/DSC00724.jpg",
          "library/DSC00725.jpg",
          "library/DSC00726.jpg",
          "library/DSC00727.jpg",
          "library/DSC00728.jpg",
          "library/DSC00729.jpg",
        };

        for (int i = 0; i < numImages; i++)
        {
            Mat im = imread(filenames[i]);
            images.push_back(im);
        }
    }
    

}

void mycvtColor(const Mat& m, Mat& src) {
    vector<Mat> splits(3);
    split(m, splits);
    src = (splits[0] * 0.114 + splits[1] * 0.587 + splits[2] * 0.299) / 3;
}

void CPU_MERTENS_process(const vector<Mat>& pics, const vector<Mat>& gW, Mat& result) {
    auto start = chrono::system_clock::now();
    int pic_num = (int)pics.size();
    Size sz = pics[0].size();
    vector<Mat> W(pic_num);
    Mat ALL_W(sz, CV_64FC1, Scalar::all(0));
    vector<Mat> _pics(pic_num);
    for (int i = 0; i < pic_num; ++i) {
        Mat gray, C, S = Mat(sz, CV_64FC1, Scalar::all(0));
        Mat E = Mat(sz, CV_64FC1, Scalar::all(1)), diff, s;
        pics[i].convertTo(_pics[i], CV_64FC3, 1.0 / 255);
        vector<Mat> split_pic(3);
        split(_pics[i], split_pic);
        mycvtColor(_pics[i], gray);
        //contrast
        Laplacian(gray, C, CV_64FC1);
        C = abs(C);
        //saturation
        Mat mean = (split_pic[0] + split_pic[1] + split_pic[2]) / 3.0;
        
        for (int c = 0; c < 3; ++c) {
            pow(split_pic[c] - mean, 2, diff);
            S += diff;
        }
        sqrt(S / 3.0, S);
        //well-exposured
        for (int c = 0; c < 3; ++c) {
            pow((split_pic[c] - 0.5) / (2 * sqrt(0.2)), 2, s);
            exp(-s, s);
            E = E.mul(s);
        }
        W[i] = Mat(sz, CV_64FC1, Scalar::all(1));
        pow(C, _para[0], C);
        pow(S, _para[1], S);
        pow(E, _para[2], E);
        W[i] = W[i].mul(C);
        W[i] = W[i].mul(S);
        if (!gW.empty()) {
            exp(gW[i], gW[i]);
            W[i] = W[i].mul(gW[i]);
        }
        W[i] = W[i].mul(E) + 1e-20;
        ALL_W += W[i];

        /*cout  << "C: " << gray.at<double>(2907, 3625) << endl;
        cout << "S: " << S.at<double>(2907, 3625) << endl;
        cout << "E: " << E.at<double>(2907, 3625) << endl;
        cout << "--------------------------------" << endl;*/
    }

    /*for (int i = 0; i < pic_num; i++)
        cout << i << ": " << W[i].at<double>(2907, 3625) << endl;*/
    

    Mat up;
    vector<Mat> final_pics(_para[3] + 1);
    vector< vector<Mat> > pics_pyr(pic_num), W_pyr(pic_num);
    for (int i = 0; i < pic_num; ++i) {
        W[i] /= ALL_W;
        buildPyramid(_pics[i], pics_pyr[i], _para[3]);
        buildPyramid(W[i], W_pyr[i], _para[3]);
    }
    /*for (int i = 0; i < pic_num; i++)
        cout << i << ": " << W[i].at<double>(2907, 3625) << endl;*/
    for (int i = 0; i <= _para[3]; ++i)
        final_pics[i] = Mat(pics_pyr[0][i].size(), CV_64FC3, Scalar::all(0));
    for (int i = 0; i < pic_num; ++i) {
        for (int j = 0; j < _para[3]; ++j) {
            pyrUp(pics_pyr[i][j + 1], up, pics_pyr[i][j].size());
            pics_pyr[i][j] -= up;
        }
        for (int j = 0; j <= _para[3]; ++j) {
            vector<Mat> split_pics_pyr(3);
            split(pics_pyr[i][j], split_pics_pyr);
            for (int c = 0; c < 3; ++c)
                split_pics_pyr[c] = split_pics_pyr[c].mul(W_pyr[i][j]);
            merge(split_pics_pyr, pics_pyr[i][j]);
            final_pics[j] += pics_pyr[i][j];
        }
    }
    for (int i = _para[3] - 1; i >= 0; --i) {
        pyrUp(final_pics[i + 1], up, final_pics[i].size());
        final_pics[i] += up;
    }
    result = final_pics[0].clone();
    auto end = chrono::system_clock::now();
    chrono::duration<double> t = end - start;
    cout << "run time: " << t.count() * 1000. << " ms." << endl;
}

__global__ void Color2Gray(cuda::PtrStepSz<double> R, cuda::PtrStepSz<double> G, cuda::PtrStepSz<double> B, cuda::PtrStepSz<double> gray)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % R.cols;
    int y = threadId / R.cols;

    if (y < R.rows) {
        gray(y, x) = (R(y, x) * 0.114 + G(y, x) * 0.587 + B(y, x) * 0.299) / 3.0f;
    }

    

    __syncthreads();
}

__device__ bool checkBound(int x, int y, int xbound, int ybound)
{
    return x < 0 || x >= xbound || y < 0 || y >= ybound;
}

__global__ void GPU_Mertens(cuda::PtrStepSz<uchar3> color_input, cuda::PtrStepSz<double> gray, cuda::PtrStepSz<float> C,
                            cuda::PtrStepSz<float> S, cuda::PtrStepSz<float> E,
                            cuda::PtrStepSz<double> R, cuda::PtrStepSz<double> G, cuda::PtrStepSz<double> B)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % color_input.cols;
    int y = threadId / color_input.cols;

    if (y < color_input.rows) {
        //contrast
        //lap filter
        double right, left, up, bottom;
        if (checkBound(x + 1, y, color_input.cols, color_input.rows))
            right = 0;
        else right = gray(y, x + 1);

        if (checkBound(x - 1, y, color_input.cols, color_input.rows))
            left = 0;
        else left = gray(y, x - 1);

        if (checkBound(x, y + 1, color_input.cols, color_input.rows))
            bottom = 0;
        else bottom = gray(y + 1, x);

        if (checkBound(x, y - 1, color_input.cols, color_input.rows))
            up = 0;
        else up = gray(y - 1, x);


        float answer = fabsf(right + left + up + bottom - 4.0f * gray(y, x));

        C(y, x) = answer;


        __syncthreads();

        //saturation
        //stdev for each pixel
        double tmp_r = R(y, x);
        double tmp_g = G(y, x);
        double tmp_b = B(y, x);
        double mean = (tmp_r + tmp_g + tmp_b) / 3.0f;
        float tmp_s = pow(tmp_r - mean, 2) + pow(tmp_g - mean, 2) + pow(tmp_b - mean, 2);


        S(y, x) = sqrt(tmp_s / 3.0f);


        __syncthreads();



        //Exposure
        //weight intensity based on how close to 0.5 with a Gauss curve with sigma = 0.2
        double tmp_s1 = pow((tmp_r - 0.5) / (2 * sqrt(0.2f)), 2);
        tmp_s1 = exp(-tmp_s1);

        double tmp_s2 = pow((tmp_g - 0.5) / (2 * sqrt(0.2f)), 2);
        tmp_s2 = exp(-tmp_s2);

        double tmp_s3 = pow((tmp_b - 0.5) / (2 * sqrt(0.2f)), 2);
        tmp_s3 = exp(-tmp_s3);

        answer = (tmp_s1 * tmp_s2 * tmp_s3);

        E(y, x) = answer;

        
    }
    __syncthreads();
}

__global__ void ComputeWeight(cuda::PtrStepSz<double> weight, cuda::PtrStepSz<double> weight_sum, cuda::PtrStepSz<float> C, cuda::PtrStepSz<float> S, cuda::PtrStepSz<float> E)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % weight.cols;
    int y = threadId / weight.cols;

    if (y < weight.rows) {
        weight(y, x) = C(y, x) * S(y, x) * E(y, x) + 1e-20;
        weight_sum(y, x) += weight(y, x);
    }

    __syncthreads();
        
}

__global__ void NormalizeWeight(cuda::PtrStepSz<double> weight, cuda::PtrStepSz<double> weight_sum)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % weight.cols;
    int y = threadId / weight.cols;

    if (y < weight.rows) {
        weight(y, x) /= weight_sum(y, x);
    }


    __syncthreads();
}

void compute_CSE(const cuda::GpuMat& input1, cuda::GpuMat& gray, cuda::GpuMat& C, cuda::GpuMat& S, cuda::GpuMat& E, cuda::GpuMat& W, cuda::GpuMat& All_W, 
                cuda::GpuMat& R, cuda::GpuMat& G, cuda::GpuMat& B, cudaStream_t ss)
{
    int rows = input1.rows;
    int cols = input1.cols;

    Color2Gray << <dim3(rows, 6), dim3(5, 199), 0, ss >> > (R, G, B, gray);


    GPU_Mertens << <dim3(rows, 6), dim3(5, 199), 0, ss >> > (input1, gray, C, S, E, R, G, B);


    ComputeWeight << <dim3(rows, 6), dim3(5, 199), 0, ss >> > (W, All_W, C, S, E);


}



int main()
{
	cout << "Reading images ... " << endl;
	vector<Mat> images;

    vector<Mat> W1;

    const int scene_No = 0;

	bool needsAlignment = true;

	readImages(images, scene_No);

    if (needsAlignment)
    {
        cout << "Aligning images ... " << endl;
        Ptr<AlignMTB> alignMTB = createAlignMTB();
        alignMTB->process(images, images);
    }
    else
    {
        cout << "Skipping alignment ... " << endl;
    }


    if (cpu) {
        

        cout << "Merging using Exposure Fusion ... " << endl;
        Mat exposureFusion;
        /*Ptr<MergeMertens> mergeMertens = createMergeMertens();
        mergeMertens->process(images, exposureFusion);*/

        //daikon
        CPU_MERTENS_process(images, W1, exposureFusion);

        cout << "Saving output ... exposure-fusion.jpg" << endl;
        imwrite("exposure-fusion.jpg", exposureFusion * 255);
    
    }
    else {
        //initializig
        cout << "Merging using Exposure Fusion ... " << endl;

        cuda::GpuMat img1(images[0].rows, images[0].cols, CV_32FC3);
        cuda::GpuMat grayImages( images[0].rows, images[0].cols , CV_64FC1);



        cuda::GpuMat contrast(images[0].rows, images[0].cols, CV_32FC1);
        cuda::GpuMat saturation(images[0].rows, images[0].cols, CV_32FC1);
        cuda::GpuMat exposure(images[0].rows, images[0].cols, CV_32FC1);

        cuda::GpuMat R(images[0].rows, images[0].cols, CV_64FC1);
        cuda::GpuMat G(images[0].rows, images[0].cols, CV_64FC1);
        cuda::GpuMat B(images[0].rows, images[0].cols, CV_64FC1);

        vector<Mat> cpu_Weights;

        cuda::GpuMat W(images[0].rows, images[0].cols, CV_64FC1);
        cuda::GpuMat All_W(images[0].rows, images[0].cols, CV_64FC1);

        grayImages.setTo(Scalar::all(0));
        contrast.setTo(Scalar::all(0));
        saturation.setTo(Scalar::all(0));
        exposure.setTo(Scalar::all(1));

        W.setTo(Scalar::all(0));
        All_W.setTo(Scalar::all(0));


        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        float gpu_time = 0.0f;
        cudaDeviceSynchronize();
        cudaEventRecord(start);


        // Compute Contrast, saturation, exposure
        const int num_streams = images.size();

        cudaStream_t stream[7];

        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&stream[i]);

            img1.upload(images[i]);

            images[i].convertTo(images[i], CV_64FC3, 1.0 / 255);

            vector< cuda::GpuMat > RGB;

            cuda::split(images[i], RGB);


            Mat result;

            compute_CSE(img1, grayImages, contrast, saturation, exposure, W, All_W, RGB[0], RGB[1], RGB[2], stream[i]);

            W.download(result);

            cpu_Weights.push_back(result);

            //cout << "--------------------------------" << endl;
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(stream[i]);
        }


        //for (int i = 0; i < images.size(); i++)
        //    cout << i << ": " << cpu_Weights[i].at<double>(2907, 3625) << endl;

        // Normalizeing weight matrix
        cuda::Stream stream1[7];

        //cudaStream_t streams[7];

        for (int i = 0; i < num_streams; i++) {

            //cudaStreamCreate(&streams[i]);

            W.upload(cpu_Weights[i]);

            cuda::divide(W, All_W, W, 1, -1, stream1[i]);
            //NormalizeWeight << <dim3(2999, 4), dim3(5, 199), 0, streams[i] >> > (W, All_W);
            Mat result;
            W.download(cpu_Weights[i]);

            //cpu_Weights[i] = result;

        }
        cudaDeviceSynchronize();
        /*for(int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
        }*/

        //for (int i = 0; i < images.size(); i++)
        //    cout << i << ": " << cpu_Weights[i].at<double>(2907, 3625) << endl;



        //build pyramid
        

        vector<Mat> final_pics(_para[3] + 1);
        vector< vector<Mat> > pics_pyr(images.size()), W_pyr(images.size());
        //from big to small
        
        for (int i = 0; i < num_streams; i++) {

            images[i].convertTo(images[i], CV_32FC3, 1.0);
            cpu_Weights[i].convertTo(cpu_Weights[i], CV_32FC1, 1.0);

            cuda::GpuMat bot_image(images[i]);
            cuda::GpuMat bot_weight(cpu_Weights[i]);

            pics_pyr[i].push_back(images[i]);
            W_pyr[i].push_back(cpu_Weights[i]);

            

            for (int j = 0; j < _para[3]; j++) {
                cuda::GpuMat down_image;
                cuda::GpuMat down_weight;
                cuda::pyrDown(bot_image, down_image, stream1[i]);
                cuda::pyrDown(bot_weight, down_weight, stream1[i]);

                Mat tmp, tmp2;
                down_image.download(tmp, stream1[i]);
                down_weight.download(tmp2, stream1[i]);
                pics_pyr[i].push_back(tmp);
                W_pyr[i].push_back(tmp2);


                bot_image = down_image;
                bot_weight = down_weight;
            }
            
        }
        cudaDeviceSynchronize();

        for (int i = 0; i <= _para[3]; ++i)
            final_pics[i] = Mat(pics_pyr[0][i].size(), CV_32FC3, Scalar::all(0));

        

        for (int i = 0; i < num_streams; i++) {

            
            for (int j = 0; j < _para[3]; j++) {
                
                //upsampling
                cuda::GpuMat bot_image(pics_pyr[i][j+1]);

                
                cuda::GpuMat subtractResult;
                cuda::GpuMat ex_subttract(pics_pyr[i][j]);
                cuda::GpuMat up_image(pics_pyr[i][j]);

                cuda::pyrUp(bot_image, up_image, stream1[i]);

                int pad_col = up_image.cols - ex_subttract.cols;
                int pad_row = up_image.rows - ex_subttract.rows;

                cuda::GpuMat cropped(up_image, Rect(0, 0, up_image.cols - pad_col, up_image.rows - pad_row));

                cuda::subtract(ex_subttract, cropped, subtractResult, noArray(), -1, stream1[i]);
                
                subtractResult.download(pics_pyr[i][j], stream1[i]);
            }
            for (int j = 0; j <= _para[3]; j++) {

                cuda::GpuMat tmp_W;

                tmp_W.upload(W_pyr[i][j]);

                vector< cuda::GpuMat > dst;

                cuda::split(pics_pyr[i][j], dst, stream1[i]);

                cuda::multiply(dst[0], tmp_W, dst[0], 1, -1, stream1[i]);
                cuda::multiply(dst[1], tmp_W, dst[1], 1, -1, stream1[i]);
                cuda::multiply(dst[2], tmp_W, dst[2], 1, -1, stream1[i]);

                cuda::merge(dst, pics_pyr[i][j], stream1[i]);

                final_pics[j] += pics_pyr[i][j];
            }

        }

        //Mat tmp3, tmp4;
        cuda::Stream stream2[8];
        for (int i = _para[3] - 1; i >= 0; --i) {
            cuda::GpuMat up_image;

            cuda::GpuMat bot_image(final_pics[i + 1]);
            cuda::GpuMat ex_add(final_pics[i]);
            cuda::pyrUp(bot_image, up_image, stream2[7 - i]);
            cuda::GpuMat addResult;

            int pad_col = up_image.cols - ex_add.cols;
            int pad_row = up_image.rows - ex_add.rows;

            cuda::GpuMat cropped(up_image, Rect(0, 0, up_image.cols - pad_col, up_image.rows - pad_row));

            cuda::add(ex_add, cropped, addResult, noArray(), -1, stream2[7 - i]);

            addResult.download(final_pics[i], stream2[7 - i]);

        }

        



        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpu_time, start, end);
        printf("\ncuda exposure fusion call: %.4f ms\n", gpu_time);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        Mat tmp4;
        tmp4 = final_pics[0].clone();

        imwrite("exposure-fusion_gpu.jpg", tmp4 * 255);
 
    }

	return 0;
}