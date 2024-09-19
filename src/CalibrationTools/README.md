# calibration_tools

把autoware的传感器校准包给独立出来了，不需要依赖autoware就可以运行
Calibration tools used for autonomous driving

## Requirement

- Ubuntu22.04
- Ros Humble

## complie
``` bash
pip install -r src/CalibrationTools/requirements.txt
colcon build --packages-select intrinsic_camera_calibrator #相机内参
colcon build --packages-select interactive_camera_lidar_calibrator #camera与lidar外参
```


## Implemented calibration tools

### sensor

We provide calibration tool for sensor pairs like LiDAR - LiDAR, LiDAR - Camera, etc.

[README](sensor/README.md)

### system - tunable static tf broadcaster

GUI to modify the parameters of generic TFs.

[README](system/tunable_static_tf_broadcaster/README.md)


>https://github.com/tier4/CalibrationTools