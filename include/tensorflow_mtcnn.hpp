#ifndef __TENSORFLOW_MTCNN_HPP__
#define __TENSORFLOW_MTCNN_HPP__

#include "tensorflow/c/c_api.h"
#include "mtcnn.hpp"
#include "comm_lib.hpp"

class tf_mtcnn: public mtcnn {

	public:
		tf_mtcnn()=default;

		int load_model(const std::string& model_dir);

		void detect(std::vector<cv::Mat>& imgs, std::vector<FACEBOXES>& face_lists);

		~tf_mtcnn();


	protected:


		void run_PNet(const std::vector<cv::Mat>& imgs, 
				scale_window& win, 
				std::vector<FACEBOXES>& box_lists);


		void run_RNet(const std::vector<cv::Mat>& imgs,
				std::vector<FACEBOXES>& pnet_boxess, 
				std::vector<FACEBOXES>& output_boxes);

		void run_ONet(const std::vector<cv::Mat>& imgs,
				std::vector<FACEBOXES>& rnet_boxess, 
				std::vector<FACEBOXES>& output_boxess);
	private:

		TF_Session * sess_;
		TF_Graph  *  graph_;

};


#endif
