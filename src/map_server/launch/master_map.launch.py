from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    start_map_saver_server_cmd = launch_ros.actions.Node(
        package = 'map_server',
        executable = 'master_map_executable',
        parameters = [
          {'json_filename':'/home/dyf/project/lidar_camera_perception_ws/src/map_server/data'},
          # {'map_name': '488811cc-79ac-4270-95fe-c72ed49fea5a'},
          {'topic_name': 'master_map'},
          {'frame_id':'odom'}])
    
    start_lifecycle_manager_cmd = launch_ros.actions.Node(
        package = 'nav2_lifecycle_manager',
        executable = 'lifecycle_manager',
        parameters = [
          {'use_sim_time': False},
          {'autostart': True},
          {'node_names': ['map_server']}])
    ld = LaunchDescription()
    ld.add_action(start_map_saver_server_cmd)
    ld.add_action(start_lifecycle_manager_cmd)
    return ld