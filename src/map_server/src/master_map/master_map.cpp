#include "master_map/master_map.h"

#include <string>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "jsoncpp/json/json.h"
#include "lifecycle_msgs/msg/state.hpp"
#include "Magick++.h"

using namespace std::placeholders;

namespace ubt_map_server{

MasterMap::MasterMap(const rclcpp::NodeOptions& options)
: nav2_util::LifecycleNode("map_server", "", options), map_available_(false)
{
  RCLCPP_INFO(this->get_logger(), "Creating");

  declare_parameter("json_filename", "");
  declare_parameter("topic_name", "master_map");
  declare_parameter("frame_id", "map");
  declare_parameter("load_master_map", "load_master_map");
}

MasterMap::~MasterMap(){}

nav2_util::CallbackReturn
MasterMap::on_configure(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Configuring");

  json_filename_ = get_parameter("json_filename").as_string();

  if (json_filename_.empty()) {
    RCLCPP_INFO(
      this->get_logger(),
      "json filename parameter is empty");
    return nav2_util::CallbackReturn::FAILURE;
  }

  std::string topic_name = get_parameter("topic_name").as_string();
  if(topic_name.empty()){
    topic_name = "master_map";
  }
  std::string frame_id =  get_parameter("frame_id").as_string();
  if(frame_id.empty()){
    frame_id = "map";
  }

  std::string load_map_service_name = get_parameter("load_master_map").as_string();
  if(load_map_service_name.empty()){
    load_map_service_name = "load_master_map";
  }

  const std::string service_prefix = get_name() + std::string("/");

  load_map_service_ = create_service<map_server::srv::LoadMasterMap>(
    service_prefix + std::string(load_map_service_name),
    std::bind(&MasterMap::loadMapCallback, this, _1, _2, _3));

  costmap_pub_ = create_publisher<nav2_msgs::msg::Costmap>(
    service_prefix + std::string(topic_name),
    rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
MasterMap::on_activate(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Activating");
  costmap_pub_->on_activate();
  if (map_available_) {
    publishCostMap();
  }
  createBond();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
MasterMap::on_deactivate(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating");
  costmap_pub_->on_deactivate();
  destroyBond();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
MasterMap::on_cleanup(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up");
  costmap_pub_.reset();
  load_map_service_.reset();
  map_available_ = false;
  occ_ = nav_msgs::msg::OccupancyGrid();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
MasterMap::on_shutdown(const rclcpp_lifecycle::State & /*state*/)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down");
  return nav2_util::CallbackReturn::SUCCESS;
}

void MasterMap::publishCostMap(){
    costmap_ = nav2_costmap_2d::Costmap2D(occ_);
    auto costmap_msg = std::make_unique<nav2_msgs::msg::Costmap>();
    costmap_msg->metadata.size_x = costmap_.getSizeInCellsX();
    costmap_msg->metadata.size_y = costmap_.getSizeInCellsY();
    costmap_msg->metadata.resolution = costmap_.getResolution();
    costmap_msg->metadata.origin = occ_.info.origin;
    costmap_msg->metadata.layer = "master_map";
    costmap_msg->data.resize(costmap_.getSizeInCellsX() * costmap_.getSizeInCellsY());
    unsigned char* char_map = costmap_.getCharMap();
    std::copy(char_map, char_map + costmap_msg->data.size(), costmap_msg->data.begin());
    costmap_msg->header= occ_.header;
    costmap_msg->header.stamp = this->now();
    costmap_pub_->publish(std::move(costmap_msg));
}

void MasterMap::loadMapCallback(
    const std::shared_ptr<rmw_request_id_t>/*request_header*/,
    const std::shared_ptr<map_server::srv::LoadMasterMap::Request> request,
    std::shared_ptr<map_server::srv::LoadMasterMap::Response> response){
    if (get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      RCLCPP_WARN(
        this->get_logger(),
        "Received LoadMap request but not in ACTIVE state, ignoring!");
      response->is_success = false;
      response->status = "NOT ACTIVE";
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Handling LoadMap request");
    if(load(request->map_name, response) == true){
      response->is_success = true;
      response->status = "LOAD_MASPTER_MAP_SUCCESS";
      map_available_ = true;
      publishCostMap();
      RCLCPP_INFO(this->get_logger(), "LoadMap request handled successfully");
    }
  }



bool MasterMap::load(const std::string & map_name,
    std::shared_ptr<map_server::srv::LoadMasterMap::Response> response){
  switch (loadMasterMapFromJson(map_name)) {
    case MASPTER_MAP_DOES_NOT_EXIST:
      response->is_success = false;
      response->status = "MASPTER_MAP_DOES_NOT_EXIST";
      return false;
    case INVALID_MASPTER_MAP_METADATA:
      response->is_success = false;
      response->status = "INVALID_MASPTER_MAP_METADATA";
      return false;
    case INVALID_MASPTER_MAP_DATA:
      response->is_success = false;
      response->status = "INVALID_MASPTER_MAP_DATA";
      return false;
    case LOAD_MASPTER_MAP_SUCCESS:
      response->is_success = true;
      response->status = "LOAD_MASPTER_MAP_SUCCESS";
      return true;
  }
  return false;
}

LOAD_MASTER_MAP_STATUS MasterMap::loadMasterMapFromJson(const std::string& map_name){
  std::string map_info = json_filename_ + "/" + map_name + "/" + map_name + ".json";
  std::string map_png = json_filename_ + "/" + map_name + "/map.png";
  std::ifstream json_file(map_info, std::ifstream::binary);
  if(!json_file.is_open()){
    RCLCPP_ERROR(this->get_logger(), "json file  %s does not exist, can't load!", map_info.c_str());
    return INVALID_MASPTER_MAP_METADATA;
  }

  Json::Value json_data;
  json_file >> json_data;
  json_file.close();
  if(json_data.empty()){
    RCLCPP_ERROR(this->get_logger(), "json file is empty, can't load!");
    return INVALID_MASPTER_MAP_METADATA;
  }

  const Json::Value& mapInfo = json_data["mapInfo"];
  int height = mapInfo["height"].asInt();
  int width = mapInfo["width"].asInt();
  std::vector<float> origin;
  for(int i = 0; i < 2; i++){
    origin.emplace_back(mapInfo["origin"][i].asFloat());
  }
  double resolution = mapInfo["resolution"].asFloat();
  // double occ_threshold = mapInfo["occ_threshold"].asFloat();
  // double free_threshold = mapInfo["free_threshold"].asFloat();
  // double negate = mapInfo["negate"].asFloat();
  try{
    Magick::InitializeMagick(nullptr);
    Magick::Image image(map_png);
    occ_.info.width = width;
    occ_.info.height = height;
    occ_.info.resolution = resolution;
    occ_.info.origin.position.x = origin[0];
    occ_.info.origin.position.y = origin[1];
    occ_.info.origin.position.z = origin[2];
    occ_.info.origin.orientation.x = 0;
    occ_.info.origin.orientation.y = 0;
    occ_.info.origin.orientation.z = 0;
    occ_.info.origin.orientation.w = 1;
    occ_.data.resize(occ_.info.width * occ_.info.height);
    for(size_t y=0;y < occ_.info.height;y++){
      for(size_t x=0; x < occ_.info.width;x++){
        auto pixel = image.pixelColor(x, y);
        std::vector<Magick::Quantum> channels = 
        { pixel.redQuantum(), 
          pixel.greenQuantum(), 
          pixel.blueQuantum()};
        if(image.matte()){
          channels.push_back(pixel.alphaQuantum());
        }
        double sum = 0;
        for(auto& channel: channels){
          sum += channel;
        }
        double shade = Magick::ColorGray::scaleQuantumToDouble(sum / channels.size());
        double occ = 1 - shade;
        int8_t map_cell;
        if(occ > 0.65){
          map_cell = 100;
        }else if(occ < 0.196){
          map_cell = 0;
        }else{
          map_cell = -1;
        }
        occ_.data[occ_.info.width*(occ_.info.height - y - 1) + x] = map_cell;
      }
    }
  }catch(Magick::Error &error){
    RCLCPP_ERROR(this->get_logger(), "image file does not exist, can't load!");
    return INVALID_MASPTER_MAP_DATA;
  }

  return LOAD_MASPTER_MAP_SUCCESS;
}
} // namespace ubt_map_server


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ubt_map_server::MasterMap>();
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}