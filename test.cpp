/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "mtcnn.hpp"
#include "utils.hpp"
#include <sys/time.h>
static uint64_t gtm(){
	struct timeval tm;
	gettimeofday(&tm, 0);
	uint64_t re = ((uint64_t)tm.tv_sec)*1000*1000 + tm.tv_usec;
	return re;
}
static std::string to6dStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%06d", in);
	std::string re(chr);
	return re;
}
static std::string toStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%d", in);
	std::string re(chr);
	return re;
}

int main(int argc, char * argv[])
{

    mtcnn * p_mtcnn = mtcnn_factory::create_detector("tensorflow");

    if (p_mtcnn == nullptr) {
        std::cerr << "error, is not supported" << std::endl;
        std::cerr << "supported types: ";
        std::vector<std::string> type_list = mtcnn_factory::list();

        for (unsigned int i = 0; i < type_list.size(); i++)
            std::cerr << " " << type_list[i];

        std::cerr << std::endl;

        return 1;
    }

    p_mtcnn->load_model("./models");
#if 1
	int count = 4;
	int LIMIT = 150;
	for(int mm = 1; mm < LIMIT; mm++){
		std::vector<cv::Mat> frames;

		for(int i = 0; i < count; i++){
			//mm = 117;
			std::string path = "/home/xyz/code1/xyz/img1/";	
			path += to6dStr(mm+i*LIMIT);
#else
	int count = 1;
	for(int mm = 1; mm < 300; mm++){
		std::vector<cv::Mat> frames;

		for(int i = 0; i < count; i++){
			std::string path = "/home/xyz/code1/xyz/img1/";	
			path += to6dStr(mm+300);

#endif
			path += ".jpg";
			cv::Mat frame1 = cv::imread(path);		
			frames.push_back(frame1);
		}

		uint64_t tm1 = gtm();
		printf("11111\n");
    		std::vector<std::vector<face_box>> face_infos;
    		p_mtcnn->detect(frames,face_infos);
		uint64_t tm2 = gtm();
		printf("22222, costtime:%lu\n", tm2-tm1);

		for(int ii = 0; ii < count; ii++){
			std::vector<face_box> &face_info = face_infos[ii];
			cv::Mat &frame = frames[ii];

    			for(unsigned int i = 0; i < face_info.size(); i++) {
        			face_box& box = face_info[i];

        			printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",i,
                			box.x0,box.y0,box.x1,box.y1, box.score);
        				printf("landmark: ");

        			for(unsigned int j = 0; j < 5; j++)
            				printf(" (%2.5f %2.5f)",box.landmark.x[j], box.landmark.y[j]);

        			printf("\n");


        			/*draw box */
        			cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);

        			/* draw landmark */
        			for (int l = 0; l < 5; l++) {
            				cv::circle(frame,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(0, 0, 255),1.8);
        			}

    			}
			cv::imshow(toStr(ii), frame);
		}
		cv::waitKey(1);
		
	}
    return 0;
}
