//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include<string>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/reduce.h>
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

const int n=8;							////grid size, we will change this value to up to 256 to test your code
const int g=1;							////padding size
const int s=(n+2*g)*(n+2*g);			////array size
//#define I(i,j) (i+g)*(n+2*g)+(j+g)		////2D coordinate -> array index
//#define S(i,j) (i+g)*(8+2*g)+(j+g)		////2D coordinate -> shared memory index
//#define B(i,j) i<0||i>=n||j<0||j>=n		////check boundary
const bool cpu = false;				////set false to turn off print for x and residual

vector<double> _para = { 1, 1, 1, 8 };

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here
void readImages(vector<Mat>& images)
{

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
            pow((split_pic[c] - 0.5) / (0.2 * sqrt(2)), 2, s);
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

        cout  << "C: " << gray.at<double>(969, 3826) << endl;
        cout << "S: " << S.at<double>(969, 3826) << endl;
        cout << "E: " << E.at<double>(969, 3826) << endl;
        cout << "--------------------------------" << endl;
    }

    for (int i = 0; i < pic_num; i++)
        cout << i << ": " << W[i].at<double>(969, 3826) << endl;
    auto end = chrono::system_clock::now();
    chrono::duration<double> t = end - start;
    cout << "run time: " << t.count() * 1000. << " ms." << endl;

    Mat up;
    vector<Mat> final_pics(_para[3] + 1);
    vector< vector<Mat> > pics_pyr(pic_num), W_pyr(pic_num);
    for (int i = 0; i < pic_num; ++i) {
        W[i] /= ALL_W;
        buildPyramid(_pics[i], pics_pyr[i], _para[3]);
        buildPyramid(W[i], W_pyr[i], _para[3]);
    }
    for (int i = 0; i < pic_num; i++)
        cout << i << ": " << W[i].at<double>(969, 3826) << endl;
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
}

__global__ void Color2Gray(cuda::PtrStepSz<double> R, cuda::PtrStepSz<double> G, cuda::PtrStepSz<double> B, cuda::PtrStepSz<double> gray)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % 3980;
    int y = threadId / 3980;

    gray(y, x) = (R(y, x) * 0.114 + G(y, x) * 0.587 + B(y, x) * 0.299) / 3.0f;

    __syncthreads();
}

__device__ bool checkBound(int x, int y)
{
    return x < 0 || x >= 3980 || y < 0 || y >= 2999;
}

__global__ void GPU_Mertens(cuda::PtrStepSz<uchar3> color_input, cuda::PtrStepSz<double> gray, cuda::PtrStepSz<double> C,
                            cuda::PtrStepSz<double> S, cuda::PtrStepSz<double> E,
                            cuda::PtrStepSz<double> R, cuda::PtrStepSz<double> G, cuda::PtrStepSz<double> B)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % 3980;
    int y = threadId / 3980;



    //contrast
    double right, left, up, bottom;
    if (checkBound(x + 1, y))
        right = 0;
    else right = gray(y, x + 1);

    if (checkBound(x - 1, y))
        left = 0;
    else left = gray(y, x - 1);

    if (checkBound(x , y+1))
        bottom = 0;
    else bottom = gray(y+1, x);

    if (checkBound(x, y-1))
        up = 0;
    else up = gray(y-1, x);

    
    /*double tmp = fabsf(double(lap(y, x)) );*/

    C(y, x) = fabsf(right + left + up + bottom - 4.0f*gray(y, x));

    

    //C(y, x) = lap(y, x);
    __syncthreads();

    //saturation
    double tmp_r = R(y, x);
    double tmp_g = G(y, x);
    double tmp_b = B(y, x);
    double mean = (tmp_r + tmp_g + tmp_b) / 3.0f;
    double tmp_s = pow(tmp_r - mean, 2) + pow(tmp_g - mean, 2) + pow(tmp_b - mean, 2);
    
    S(y, x) = sqrt(tmp_s / 3.0f);
    

    __syncthreads();

    /*if (S(y, x) == 0.0f)
        S(y, x) = sqrt(tmp_s / 3.0f);
    __syncthreads();*/

    if (sqrt(tmp_s / 3.0f) != S(y, x)) {
        printf("S1: %d, S2: %d\n", x, y);
    }


    if (threadId == 3860446) {
        /*printf("mean: %f\n", mean);
        printf("R: %f\n", tmp_r);
        printf("G: %f\n", tmp_g);
        printf("B: %f\n", tmp_b);*/
        /*printf("S1: %f\n", sqrt(tmp_s / 3.0f));
        printf("S2: %f\n", S(y, x));*/
        /*if (sqrt(tmp_s / 3.0f) != S(y, x))
            printf("S1: %f\n", sqrt(tmp_s / 3.0f));*/
    }

    

    //Exposure
    double tmp_s1 = pow((tmp_r - 0.5) / (0.2 * sqrt(2.0f)), 2);
    tmp_s1 = exp(-tmp_s1);

    double tmp_s2 = pow((tmp_g - 0.5) / (0.2 * sqrt(2.0f)), 2);
    tmp_s2 = exp(-tmp_s2);

    double tmp_s3 = pow((tmp_b - 0.5) / (0.2 * sqrt(2.0f)), 2);
    tmp_s3 = exp(-tmp_s3);

    E(y, x) = (tmp_s1 * tmp_s2 * tmp_s3);

    __syncthreads();
    
    /*if (threadId == 3860446) {
        printf("mean: %f\n", mean);
        printf("C: %f\n", C(969, 3826));
        printf("S: %f\n", S(969, 3826));
        printf("E: %f\n", E(969, 3826));
    }*/
        
}

__global__ void ComputeWeight(cuda::PtrStepSz<double> weight, cuda::PtrStepSz<double> weight_sum, cuda::PtrStepSz<double> C, cuda::PtrStepSz<double> S, cuda::PtrStepSz<double> E)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % 3980;
    int y = threadId / 3980;

    weight(y, x) = C(y, x) * S(y, x) * E(y, x) + 1e-20;
    weight_sum(y, x) += weight(y, x);

    __syncthreads();

    /*if (threadId == 1) {
        printf("C: %f\n", C(969, 3826));
        printf("S: %f\n", S(969, 3826));
        printf("E: %f\n", E(969, 3826));
        printf("weight: %g\n", weight(969, 3826));
        printf("sum: %g\n", weight_sum(0, 1));
    }*/
        
}

__global__ void NormalizeWeight(cuda::PtrStepSz<double> weight, cuda::PtrStepSz<double> weight_sum)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int x = threadId % 3980;
    int y = threadId / 3980;

    weight(y, x) /= weight_sum(y, x);
    __syncthreads();
}

void compute_CSE(const cuda::GpuMat& input1, cuda::GpuMat& gray, cuda::GpuMat& C, cuda::GpuMat& S, cuda::GpuMat& E, cuda::GpuMat& W, cuda::GpuMat& All_W, 
                cuda::GpuMat& R, cuda::GpuMat& G, cuda::GpuMat& B, cudaStream_t ss)
{
    

    Color2Gray << <dim3(2999, 4), dim3(5, 199), 0, ss >> > (R, G, B, gray);


    GPU_Mertens << <dim3(2999, 4), dim3(5, 199), 0, ss >> > (input1, gray, C, S, E, R, G, B);


    ComputeWeight << <dim3(2999, 4), dim3(5, 199), 0, ss >> > (W, All_W, C, S, E);


}



int main()
{
	cout << "Reading images ... " << endl;
	vector<Mat> images;

    vector<Mat> W1;

	bool needsAlignment = true;

	readImages(images);

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
            vector<Mat> split_pic(3);
            split(images[i], split_pic);

            

            R.upload(split_pic[0]);
            G.upload(split_pic[1]);
            B.upload(split_pic[2]);

            Mat result;

            compute_CSE(img1, grayImages, contrast, saturation, exposure, W, All_W, R, G, B, stream[i]);

            W.download(result);

            cpu_Weights.push_back(result);

            //cout << "--------------------------------" << endl;
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(stream[i]);
        }


        for (int i = 0; i < images.size(); i++)
            cout << i << ": " << cpu_Weights[i].at<double>(969, 3826) << endl;

        // Normalizeing weight matrix
        cudaStream_t streams[7];

        for (int i = 0; i < num_streams; i++) {

            cudaStreamCreate(&streams[i]);

            W.upload(cpu_Weights[i]);

            // launch one worker kernel per stream
            NormalizeWeight << <dim3(2999, 4), dim3(5, 199), 0, streams[i] >> > (W, All_W);
            Mat result;
            W.download(result);

            cpu_Weights[i] = result;

        }
        cudaDeviceSynchronize();
        for(int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
        }

        for (int i = 0; i < images.size(); i++)
            cout << i << ": " << cpu_Weights[i].at<double>(969, 3826) << endl;

        //Mat up;
        //vector<Mat> final_pics(_para[3] + 1);
        //vector< vector<Mat> > pics_pyr(images.size()), W_pyr(images.size());
        //for (int i = 0; i < images.size(); ++i) {
        //    //cpu_Weights[i] /= ALL_W;
        //    buildPyramid(images[i], pics_pyr[i], _para[3]);
        //    buildPyramid(cpu_Weights[i], W_pyr[i], _para[3]);
        //}

        //build pyramid
        cuda::Stream stream1 [7];

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
                cuda::GpuMat tmp_R;
                cuda::GpuMat tmp_G;
                cuda::GpuMat tmp_B;
                cuda::GpuMat tmp_W;

                /*vector<Mat> split_pic(3);
                split(pics_pyr[i][j], split_pic);*/

                /*tmp_R.upload(split_pic[0]);
                tmp_G.upload(split_pic[1]);
                tmp_B.upload(split_pic[2]);*/
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

        Mat tmp3, tmp4;
        for (int i = _para[3] - 1; i >= 0; --i) {
            pyrUp(final_pics[i + 1], tmp3, final_pics[i].size());
            final_pics[i] += tmp3;
        }

        tmp4 = final_pics[0].clone();

        imwrite("exposure-fusion_gpu.jpg", tmp4 * 255);

        /*Mat up, tmp4;

        for (int i = 0; i <= _para[3]; ++i)
            final_pics[i] = Mat(pics_pyr[0][i].size(), CV_64FC3, Scalar::all(0));
        for (int i = 0; i < images.size(); ++i) {
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

        tmp4 = final_pics[0].clone();

        imwrite("exposure-fusion_gpu.jpg", tmp4 * 255);*/

        /*for (int i = 0; i < _para[3]; i++) {
            string name = to_string(i) + ".jpg";
            imwrite(name, pics_pyr[2][i] * 255);
        }*/

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpu_time, start, end);
        printf("\nAsync function call: %.4f ms\n", gpu_time);
        cudaEventDestroy(start);
        cudaEventDestroy(end);




 
    }
	


	if (cuda::getCudaEnabledDeviceCount() == 0)
		printf("NO CUDA\n");
	else
		printf("CUDA = %d\n", cuda::getCudaEnabledDeviceCount());

	return 0;
}