#include <iostream>
#include <vector>
#include <algorithm>
#include<cmath>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace std;



int pred(cv::Mat &image){
    //分割成三通道图像
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);

    //设定bin数目
    int histSize = 255;

    //设定取值范围
    float range[] = {0, 255};
    const float* histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    //声明三个通道的hist数组
    cv::Mat b_hist, g_hist, r_hist;

    //计算直方图
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    cv::normalize(b_hist, b_hist, 1, 0, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 1, 0, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 1, 0, cv::NORM_MINMAX);

    float feat[768];
    for (int i=0;i<256;i++){
        feat[i] = b_hist.at<float>(i, 0) ;
        feat[i+256] = g_hist.at<float>(i, 0) ;
        feat[i+512] = r_hist.at<float>(i, 0) ;
    }
    // 读取我们的权重信息
    torch::jit::script::Module module = torch::jit::load("../model.pt");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::from_blob(feat, {1,768}, at::kFloat));
    // std::cout << "inputs: " << inputs << endl;
    at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << "output: " << output << endl;

    auto res = output.accessor<float,2>();
    int res1 = (int) (res[0][0]*100);
    // std::cout << "res1: " << res1 << endl;

    if (res1 > 95){
        res1 = 95;
    }
    if (res1 < 30){
        res1 =30;
    }
    return res1;
}



int main(int argc, const char* argv[])
{
    cv::Mat srcImage = cv::imread(argv[1]);
    if(!srcImage.data)
    {
        cout << "图像加载失败!" << endl;
        return false;
    }
    std::cout <<"输入图像尺寸:"<< srcImage.rows <<" " << srcImage.cols <<" " << srcImage.channels() << std::endl;

    int DLPred = pred(srcImage);
    std::cout << "DLPred: " << DLPred << endl;

}