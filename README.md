# MTCNN C++ Implementation

This is a C++ project to implement MTCNN, fork form https://github.com/OAID/mtcnn, only for tensorflow parallel runing.

# Build

* Bulid tensorflow
   
   
   bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow/tools/lib_package:libtensorflow
	

* Install opencv package 


* TF_ROOT=/home/xyz/code1/tensorflow-1.4.0  make

# Run
If the basic work is ready (build caffe/Mxnet/Tensorflow sucessfully) followed by above steps. You can run the test now.
### 1. Test:

	./r.sh

# Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Mxnet

https://github.com/pangyupo/mxnet_mtcnn_face_detection

### MTCNN python on Tensorflow

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet

From this directory:

    facenet/src/align
