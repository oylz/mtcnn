#ifndef __COMMON_LIB_HPP__
#define __COMMON_LIB_HPP__

#define NMS_UNION 1
#define NMS_MIN  2



struct scale_window
{
	int h;
	int w;
	float scale;
};

int numpy_round(float f);

void nms_boxes(FACEBOXES& input, float threshold, int type, FACEBOXES&output);

void regress_boxes(FACEBOXES& rects);

void square_boxes(FACEBOXES& rects);

void padding(int img_h, int img_w, FACEBOXES& rects);

void process_boxes(FACEBOXES& input, int img_h, int img_w, FACEBOXES& rects);

void generate_bounding_box(const float * confidence_data, int confidence_size,
               const float * reg_data, float scale, float threshold,
               int feature_h, int feature_w, FACEBOXES&  output, bool transposed);


void set_input_buffer(std::vector<cv::Mat>& input_channels,
		float* input_data, const int height, const int width);


void  cal_pyramid_list(int height, int width, int min_size, float factor,std::vector<scale_window>& list);

void cal_landmark(FACEBOXES& box_list);

void set_box_bound(FACEBOXES& box_list, int img_h, int img_w);

#endif
