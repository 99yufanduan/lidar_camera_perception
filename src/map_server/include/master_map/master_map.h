#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav2_msgs/msg/costmap.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "map_server/srv/load_master_map.hpp"


namespace ubt_map_server{

typedef enum {
  LOAD_MASPTER_MAP_SUCCESS,
  MASPTER_MAP_DOES_NOT_EXIST,
  INVALID_MASPTER_MAP_METADATA,
  INVALID_MASPTER_MAP_DATA
} LOAD_MASTER_MAP_STATUS;

class MasterMap : public nav2_util::LifecycleNode
{
public:
  explicit MasterMap(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~MasterMap();

protected:
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;


  void loadMapCallback(
    const std::shared_ptr<rmw_request_id_t>/*request_header*/,
    const std::shared_ptr<map_server::srv::LoadMasterMap::Request> request,
    std::shared_ptr<map_server::srv::LoadMasterMap::Response> response); //从文件中载入地图
  
  bool load(const std::string & map_name,
      std::shared_ptr<map_server::srv::LoadMasterMap::Response> response);
  LOAD_MASTER_MAP_STATUS loadMasterMapFromJson(const std::string& map_name);
  void imageToOccupancyGrid(const std::string& map_png);
  void publishCostMap();

private:

  rclcpp::Service<map_server::srv::LoadMasterMap>::SharedPtr load_map_service_;
  rclcpp_lifecycle::LifecyclePublisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_pub_;

  std::string json_filename_;
  std::string frame_id_;
  nav_msgs::msg::OccupancyGrid occ_;
  nav2_costmap_2d::Costmap2D costmap_;
  bool map_available_;

};
  
} // namespace ubt_map_server