#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat image;
cv::Mat img_src;


cv::Rect box;
bool drawing_box = false;
bool is_loop=true;

void draw_box(cv::Mat* img, cv::Rect rect){
    cv::rectangle(*img, cv::Point2d(box.x, box.y), cv::Point2d(box.x + box.width, box.y + box.height),
                  cv::Scalar(128,0,0),-1, CV_AA);
}

double dist(double x1,double y1, double x2, double y2){
    return  pow( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1), 0.5 );
}


void mouseEvent(int event, int x, int y, int flags, void* param) {
    
    cv::Mat* image = static_cast<cv::Mat*>(param);
    
    
    switch (event){
        case cv::EVENT_MOUSEMOVE:
            if (drawing_box){
                box.width = x - box.x;
                box.height = y - box.y;
            }
            break;
        case cv::EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = cv::Rect(x, y, 0, 0);
            break;
            
        case cv::EVENT_LBUTTONUP:
            drawing_box = false;
            if (box.width < 0){
                box.x += box.width;
                box.width *= -1;
            }
            if (box.height < 0){
                box.y += box.height;
                box.height *= -1;
            }
            draw_box(image, box);
            break;
    }
    
}

void face(bool mosaic){
    
    cv::imwrite("output.jpg", image);
    cv::Mat output = cv::imread("output.jpg");
    
    
    cv::CascadeClassifier cascade;
    cv::CascadeClassifier cascade_anime;
    
    cascade.load("haarcascade_frontalface_alt.xml"); // XML ファイルから読み込む
    cascade_anime.load("lbpcascade_animeface.xml"); // XML ファイルから読み込む
    
    
    // 顔検出の処理
    std::vector<cv::Rect> faces[2];
    cascade.detectMultiScale(output, faces[0], 1.1, 3, 0, cv::Size(20, 20));
    cascade_anime.detectMultiScale(image, faces[1], 1.1, 3, 0, cv::Size(20, 20));
    
    int margin = 10;
    
    
    // 検出結果の描画
    for(int j=0; j<2 ; j++ ){
        for (int i = 0; i < faces[j].size(); i++){
            
            cv::Point start = cv::Point(faces[j][i].x-margin, faces[j][i].y-margin);
            cv::Point size  = cv::Point(faces[j][i].width+margin, faces[j][i].height+margin);
            cv::Point end   = cv::Point(faces[j][i].x+size.x, faces[j][i].y+size.y);
            
            //cv::rectangle(image, faces[j][i], cv::Scalar(255, 0, 0), 3);
            //cv::rectangle( image, start,end, cv::Scalar(0, 255, 0),2, CV_AA);
            
            
            
            if(mosaic){
                cv::Rect rect_cut(start.x,start.y,size.x,size.y);
                cv::Mat img_cut = img_src(rect_cut);
                cv::blur(img_cut, img_cut, cv::Size(30,30));
            }
            else{
                cv::rectangle( image, start,end, cv::Scalar(0, 255, 0),2, CV_AA);
            }
        }
    }
    
    cv::imshow("face_mosaic", img_src);
    
}


//静止画用
int main() {
    box = cv::Rect(-1, -1, 0, 0);
    
    //画像読み込み
    
    img_src = cv::imread("myu.jpg");
    image = cv::imread("myu.jpg");
    cv::Mat temp = image.clone();
    
    //各種準備
    cv::namedWindow("face_mosaic", CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback("face_mosaic", mouseEvent, (void *)&image);
    
    //    face(false);
    
    
    // Main loop
    while (is_loop){
        
        image.copyTo(temp);
        
        if (drawing_box) {
            draw_box(&temp, box);
        }
        
        cv::imshow("face_mosaic", temp);
        
        
        if (cv::waitKey(15) == 27){
            face(true);
            is_loop = false;
        }
    }
    
    cv::imwrite("result.jpg", img_src);
    cv::waitKey();
    return 0;
}







//動画用
/**
int main() {
    
    cv::VideoCapture capture("face.mp4"); // ビデオキャプチャの準備
 
    
    
    cv::Mat image; // 画像データを入れる用の変数
    cv::Mat img_resize;
    cv::namedWindow("video"); // ウィンドウの作成
    
    
    
    // 物体認識のための仕組み（カスケード型識別器）を準備する
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml"); // XML ファイルから読み込む
    
    int margin = 10;
    
    box = cv::Rect(-1, -1, 0, 0);
    
    
    cv::Mat temp = img_resize.clone();
    cv::setMouseCallback("video", mouseEvent, (void *)&image);

    
    while (true) {
        
        capture >> image; // キャプチャから画像データを読み込む
        
        
        cv::resize(image,img_resize,cv::Size(1920/3,1080/3),0, 0, cv::INTER_LINEAR);
        
        // 顔検出の処理
        std::vector<cv::Rect> faces;
        cascade.detectMultiScale(img_resize, faces, 1.1, 3, 0, cv::Size(20, 20));
        // 検出結果の描画
        for (int i = 0; i < faces.size(); i++){
            
            cv::rectangle(img_resize, faces[i], cv::Scalar(0, 0, 255), 3);
            
            cv::Point start = cv::Point(faces[i].x-margin, faces[i].y-margin);
            cv::Point size  = cv::Point(faces[i].width+margin, faces[i].height+margin);
            cv::Point end   = cv::Point(faces[i].x+size.x, faces[i].y+size.y);
            cv::Point center = cv::Point(start.x + size.x/2,start.y + size.y/2);
            
            if(box.x <= center.x && center.x <= box.x+box.width &&
               box.y <= center.y && center.y <= box.y+box.height){
                break;
            }
            
            cv::Rect rect_cut(start.x,start.y,size.x,size.y); // x,y,w,h
            cv::Mat img_cut = img_resize(rect_cut);
            cv::blur(img_cut, img_cut, cv::Size(30,30));
            
        }
        
        img_resize.copyTo(temp);
        
        if (drawing_box) {
            draw_box(&temp, box);
        }
        
        cv::imshow("video", temp); // 画像をウィンドウに表示
        int key = cv::waitKey(30);
    }
    
}
**/
