tensorflow-yolo-v3:darknet框架的权重转tensorflow框架pb格式
cd /home/xs/tensorflow_tools/tensorflow-yolo-v3
标签保存和模型名字一致：yolo_v3.labels
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/bus_station_6094_960x540.mp4
python3 demo.py --weights_file yolov3.weights --class_names coco.names --input_img Traffic.jpg --output_img out.jpg
模型格式转换：ubuntu16.04+openVINO(R4)(GPU版本,CPU去掉--data_type=FP16)
cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer
sudo python3 mo_tf.py --input_model /home/xs/xs/tensorflow-yolo-v3/yolo_v3.pb --tensorflow_use_custom_operations_config extensions/front/tf/yolo_v3.json --input_shape=[1,416,416,3] --data_type=FP16
将生成的yolo_v3.xml和yolo_v3.bin复制到本文件夹下
视频测试：
cd /home/xs/inference_engine_samples/intel64/Release
./object_detection_demo_yolov3_async -i /home/xs/tensorflow_tools/tensorflow-yolo-v3/bus_station_6094_960x540.mp4 -m /home/xs/tensorflow_tools/tensorflow-yolo-v3/yolo_v3.xml -d CPU
摄像头测试：
./object_detection_demo_yolov3_async -i cam -m /home/xs/tensorflow_tools/tensorflow-yolo-v3/yolo_v3.xml -d CPU

