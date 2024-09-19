from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import LoadComposableNodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value='/home/dyf/project/lidar_camera_perception_ws/src/occupancy_grid_map_to_cost_map/config/config.yaml',  # 替换为实际 YAML 文件路径
            description='YAML config file path'
        ),
        Node(
            package='occupancy_grid_map_to_cost_map',
            executable='occupancy_grid_map_to_cost_map_publish',
            name='pcd_filter_node',
            parameters=[LaunchConfiguration('config_file')]
        )
    ])