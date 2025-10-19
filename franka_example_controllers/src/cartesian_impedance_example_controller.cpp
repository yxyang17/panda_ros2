#include "franka_example_controllers/cartesian_impedance_example_controller.hpp"
#include "franka_example_controllers/pseudo_inversion.hpp"

namespace franka_example_controllers {

const double max_translational_stiffness = 1000.0;
const double max_translational_damping = 200.0;
const double max_rotational_stiffness = 150.0;
const double max_rotational_damping = 30.0;
const double max_nullspace_stiffness = 50.0;
const double max_translational_clip = 0.1;
const double max_rotational_clip = 0.15;
const std::string name_arm_id = "arm_id";
const std::string name_translational_stiffness = "translational_stiffness";
const std::string name_translational_damping = "translational_damping";
const std::string name_rotational_stiffness = "rotational_stiffness";
const std::string name_rotational_damping = "rotational_damping";
const std::string name_nullspace_stiffness = "nullspace_stiffness";
const std::string name_translational_clip = "translational_clip";
const std::string name_rotational_clip = "rotational_clip";
// const std::string name_q_limit = "joint_nullspace_error_limit";
const std::string name_enable_nullspace_joints = "enable_nullspace_joints";
const std::string name_q_d_nullspace = "q_d_nullspace";

void declare_double_parameter(std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_ptr,
                              const std::string& name,
                              const std::string& description,
                              double default_value,
                              double from_value,
                              double to_value);

// Find matching key vector in columns of haystack and return the best-aligned column index.
// Only consider available indexes (> 0). If no match is found, return largest available index.
static Eigen::Index findMatching(const Eigen::VectorXd& key,
                                 const Eigen::MatrixXd& haystack,
                                 const Eigen::VectorXi& available);

CartesianImpedanceExampleController::CallbackReturn CartesianImpedanceExampleController::on_init() {
  try {
    auto_declare<std::string>(name_arm_id, "panda");
    auto_declare<double>(name_translational_stiffness, 500.0);
    auto_declare<double>(name_translational_damping, 100.0);
    auto_declare<double>(name_rotational_stiffness, 60.0);
    auto_declare<double>(name_rotational_damping, 6.0);
    auto_declare<double>(name_nullspace_stiffness, 20.0);
    auto_declare<double>(name_translational_clip, 0.06);
    auto_declare<double>(name_rotational_clip, 0.1);
    auto_declare<bool>(name_enable_nullspace_joints, false);
    auto_declare<std::vector<double>>(name_q_d_nullspace, {});
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  // Equilibrium pose subscription
  sub_equilibrium_pose_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/equilibrium_pose", 20,
      std::bind(&CartesianImpedanceExampleController::equilibriumPoseCallback, this,
                std::placeholders::_1));
  // Nullspace exploration direction subscription
  sub_nullspace_dir_ = get_node()->create_subscription<std_msgs::msg::Float32>(
      "~/nullspace_direction", 20,
      std::bind(&CartesianImpedanceExampleController::nullspaceDirCallback, this,
                std::placeholders::_1));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  // Retrieve parameters and clip them
  configure_parameters();

  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::command_interface_configuration() const {
  // Define command interfaces
  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // get command config
  for (int i = 1; i <= num_joints; i++) {
    command_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return command_interfaces_config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::state_interface_configuration() const {
  // Define state interfaces
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Creates state interface for robot state
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_model_name);
  }
  // Creates state interface for robot model
  for (const auto& franka_robot_state_name : franka_robot_state_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_state_name);
  }
  return state_interfaces_config;
}

CartesianImpedanceExampleController::CallbackReturn
CartesianImpedanceExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  auto robot_description = get_node()->get_parameter("robot_description").as_string();
  franka_robot_state_ = std::make_unique<franka_semantic_components::FrankaRobotState>(
      franka_semantic_components::FrankaRobotState(arm_id_ + "/" + k_robot_state_interface_name,
                                                   robot_description));
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                   arm_id_ + "/" + k_robot_state_interface_name));
  return CallbackReturn::SUCCESS;
}

CartesianImpedanceExampleController::CallbackReturn
CartesianImpedanceExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_state_->assign_loaned_state_interfaces(state_interfaces_);
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  init_robot_state_ = franka_msgs::msg::FrankaRobotState();
  franka_robot_state_->get_values_as_message(init_robot_state_);

  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(init_robot_state_.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_robot_state_.o_t_ee.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = initial_transform.rotation();
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = initial_transform.rotation();

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianImpedanceExampleController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_state_->release_interfaces();
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type CartesianImpedanceExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/
) {
  // get state variables
  robot_state_ = franka_msgs::msg::FrankaRobotState();
  franka_robot_state_->get_values_as_message(robot_state_);

  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);

  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_j_d(robot_state_.tau_j_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state_.o_t_ee.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> position_error;
  position_error.head(3) << position - position_d_;
  // clip position error
  position_error.head(3) << position_error.head(3).cwiseMax(-translational_clip_);
  position_error.head(3) << position_error.head(3).cwiseMin(translational_clip_);

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  position_error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  position_error.tail(3) << -transform.rotation() * position_error.tail(3);
  // clip orientation error
  position_error.tail(3) << position_error.tail(3).cwiseMax(-rotational_clip_);
  position_error.tail(3) << position_error.tail(3).cwiseMin(rotational_clip_);

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  Eigen::MatrixXd jacobian_transpose_pinv;
  jacobian_transpose_pinv = pseudoInverse(jacobian.transpose(), true);

  // Cartesian PD control with damping ratio = 1
  tau_task << jacobian.transpose() *
                  (-cartesian_stiffness_ * position_error - cartesian_damping_ * (jacobian * dq));
  // update q_d_nullspace_ if nullspace exploration is enabled
  updateNullspaceExploration(jacobian);

  // Nullspace PD control with damping ratio = 1. Nullspace param would overwrite nullspace
  // exploration topic.
  if (enable_nullspace_joints_ || enable_nullspace_joints_explore_) {
    Eigen::Matrix<double, 7, 1> q_error =
        enable_nullspace_joints_ ? q_d_nullspace_ - q : q_d_nullspace_explore_error_;
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                         (nullspace_stiffness_ * q_error - (2.0 * sqrt(nullspace_stiffness_)) * dq);
    tau_d << tau_task + coriolis + tau_nullspace;
  } else {
    tau_d << tau_task + coriolis;
  }

  // saturate the commanded torque to joint limits
  tau_d << saturateTorqueRate(tau_d, tau_j_d);
  tau_d << saturateTorque(tau_d);

  for (int i = 0; i < num_joints; i++) {
    command_interfaces_[i].set_value(tau_d[i]);
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;

  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);

  return controller_interface::return_type::OK;
}

/**
 * @brief Callback function to update the target equilibrium pose.
 *
 * This method listens to an equilibrium pose message of type `geometry_msgs::msg::PoseStamped`
 * and updates the target position (`position_d_target_`) and orientation (`orientation_d_target_`)
 * accordingly. These targets are used to update the desired position (`position_d_`) and
 * orientation (`orientation_d_`), which the robot converges to for achieving equilibrium.
 *
 * @param msg Shared pointer to a `geometry_msgs::msg::PoseStamped` message containing the new
 * target pose.
 */

void CartesianImpedanceExampleController::equilibriumPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  std::cerr<<"\ntranslation "<< msg->pose.position.x << msg->pose.position.y<< msg->pose.position.z <<std::endl;
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

void CartesianImpedanceExampleController::nullspaceDirCallback(
    const std_msgs::msg::Float32::SharedPtr msg) {
  nullspace_explore_dir_ = std::clamp(msg->data, -1.0f, 1.0f);
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_j_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_j_d[i];
    tau_d_saturated[i] =
        tau_j_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::saturateTorque(
    const Eigen::Matrix<double, 7, 1>&
        tau_d_calculated) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    tau_d_saturated[i] = std::max(std::min(tau_d_calculated[i], tau_max_), -tau_max_);
  }
  return tau_d_saturated;
}

// Reference:
// https://github.com/moveit/moveit/blob/master/moveit_ros/visualization/motion_planning_rviz_plugin/src/motion_planning_frame_joints_widget.cpp#L377
void CartesianImpedanceExampleController::updateNullspaceExploration(
    const Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian) {
  if (std::fabs(nullspace_explore_dir_) < 1e-5) {
    enable_nullspace_joints_explore_ = false;
    return;
  }
  enable_nullspace_joints_explore_ = true;
  nullspace_jacobi_svd_.compute(jacobian, Eigen::ComputeFullV);
  Eigen::Index rank = nullspace_jacobi_svd_.rank();
  std::size_t ns_dim = nullspace_jacobi_svd_.cols() - rank;
  Eigen::MatrixXd ns(nullspace_jacobi_svd_.cols(), ns_dim);
  Eigen::VectorXi available(ns_dim);
  for (std::size_t j = 0; j < ns_dim; ++j)
    available[j] = j;
  for (size_t i = 0; i < ns_dim; ++i) {
    // Find matching null-space basis vector in previous nullspace_
    const Eigen::VectorXd& current = nullspace_jacobi_svd_.matrixV().col(rank + i);
    Eigen::Index index = findMatching(current, nullspace_base_, available);
    int sign = current(2) > 0 ? 1 : -1;
    ns.col(index).noalias() = sign * current;
    available[index] = -1;  // mark index as taken
  }
  nullspace_base_ = ns;
  q_d_nullspace_explore_error_ = nullspace_explore_dir_ * nullspace_base_.col(0);
}

void CartesianImpedanceExampleController::configure_parameters() {
  auto node_ptr = get_node();
  auto translational_stiffness = node_ptr->get_parameter(name_translational_stiffness).as_double();
  translational_stiffness = std::clamp(translational_stiffness, 0.0, max_translational_stiffness);
  auto translational_damping = node_ptr->get_parameter(name_translational_damping).as_double();
  translational_damping = std::clamp(translational_damping, 0.0, max_translational_damping);
  auto rotational_stiffness = node_ptr->get_parameter(name_rotational_stiffness).as_double();
  rotational_stiffness = std::clamp(rotational_stiffness, 0.0, max_rotational_stiffness);
  auto rotational_damping = node_ptr->get_parameter(name_rotational_damping).as_double();
  rotational_damping = std::clamp(rotational_damping, 0.0, max_rotational_damping);
  nullspace_stiffness_ = node_ptr->get_parameter(name_nullspace_stiffness).as_double();
  nullspace_stiffness_ = std::clamp(nullspace_stiffness_, 0.0, max_nullspace_stiffness);
  nullspace_stiffness_target_ = nullspace_stiffness_;
  translational_clip_ = node_ptr->get_parameter(name_translational_clip).as_double();
  translational_clip_ = std::clamp(translational_clip_, 0.0, max_translational_clip);
  rotational_clip_ = node_ptr->get_parameter(name_rotational_clip).as_double();
  rotational_clip_ = std::clamp(rotational_clip_, 0.0, max_rotational_clip);
  // q_limit_ = node_ptr->get_parameter(name_q_limit).as_double();
  // q_limit_ = std::clamp(q_limit_, 0.0, max_q_limit);
  enable_nullspace_joints_ = node_ptr->get_parameter(name_enable_nullspace_joints).as_bool();
  auto q_d_nullspace = node_ptr->get_parameter(name_q_d_nullspace).as_double_array();

  cartesian_stiffness_.setZero();
  cartesian_stiffness_.topLeftCorner(3, 3)
      << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  cartesian_stiffness_.bottomRightCorner(3, 3)
      << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  cartesian_damping_.setZero();
  cartesian_damping_.topLeftCorner(3, 3) << translational_damping * Eigen::MatrixXd::Identity(3, 3);
  cartesian_damping_.bottomRightCorner(3, 3)
      << rotational_damping * Eigen::MatrixXd::Identity(3, 3);
  cartesian_stiffness_target_ = cartesian_stiffness_;
  cartesian_damping_target_ = cartesian_damping_;
  for (int i = 0; i < num_joints; ++i) {
    q_d_nullspace_(i) = q_d_nullspace.at(i);
  }

  declare_double_parameter(node_ptr, name_translational_stiffness,
                           "Cartesian translational stiffness", translational_stiffness, 0.0,
                           max_translational_stiffness);
  declare_double_parameter(node_ptr, name_translational_damping, "Cartesian translational damping",
                           translational_damping, 0.0, max_translational_damping);
  declare_double_parameter(node_ptr, name_rotational_stiffness, "Cartesian rotational stiffness",
                           rotational_stiffness, 0.0, max_rotational_stiffness);
  declare_double_parameter(node_ptr, name_rotational_damping, "Cartesian rotational damping",
                           rotational_damping, 0.0, max_rotational_damping);
  declare_double_parameter(node_ptr, name_nullspace_stiffness, "Nullspace stiffness",
                           nullspace_stiffness_, 0.0, max_nullspace_stiffness);
  declare_double_parameter(node_ptr, name_translational_clip, "Cartesian translational error clip",
                           translational_clip_, 0.0, max_translational_clip);
  declare_double_parameter(node_ptr, name_rotational_clip, "Cartesian rotational error clip",
                           rotational_clip_, 0.0, max_rotational_clip);
  // declare_double_parameter(node_ptr, name_q_limit, "Nullspace joint error limit", q_limit_, 0.0,
  //                          max_q_limit);
  param_handle_ = node_ptr->add_on_set_parameters_callback(
      std::bind(&CartesianImpedanceExampleController::param_callback, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult CartesianImpedanceExampleController::param_callback(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto& param : parameters) {
    if (param.get_name() == name_translational_stiffness) {
      cartesian_stiffness_target_.topLeftCorner(3, 3)
          << param.as_double() * Eigen::Matrix3d::Identity();
    } else if (param.get_name() == name_translational_damping) {
      cartesian_damping_target_.topLeftCorner(3, 3)
          << param.as_double() * Eigen::Matrix3d::Identity();
    } else if (param.get_name() == name_rotational_stiffness) {
      cartesian_stiffness_target_.bottomRightCorner(3, 3)
          << param.as_double() * Eigen::Matrix3d::Identity();
    } else if (param.get_name() == name_rotational_damping) {
      cartesian_damping_target_.bottomRightCorner(3, 3)
          << param.as_double() * Eigen::Matrix3d::Identity();
    } else if (param.get_name() == name_nullspace_stiffness) {
      nullspace_stiffness_target_ = param.as_double();
    } else if (param.get_name() == name_translational_clip) {
      translational_clip_ = param.as_double();
    } else if (param.get_name() == name_rotational_clip) {
      rotational_clip_ = param.as_double();
    } else if (param.get_name() == name_enable_nullspace_joints) {
      enable_nullspace_joints_ = param.as_bool();
    } else if (param.get_name() == name_q_d_nullspace) {
      auto q_d_nullspace = param.as_double_array();
      for (int i = 0; i < num_joints; ++i) {
        q_d_nullspace_(i) = q_d_nullspace.at(i);
      }
    } else {
      result.successful = false;
      result.reason = "Parameter set behavior not defined";
    }
  }
  return result;
}

void declare_double_parameter(std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_ptr,
                              const std::string& name,
                              const std::string& description,
                              double default_value,
                              double from_value,
                              double to_value) {
  auto parameter_descriptor = rcl_interfaces::msg::ParameterDescriptor();
  auto range = rcl_interfaces::msg::FloatingPointRange();
  range.from_value = from_value;
  range.to_value = to_value;
  parameter_descriptor.floating_point_range.emplace_back(range);
  parameter_descriptor.description = description;
  if (node_ptr->has_parameter(name)) {
    node_ptr->undeclare_parameter(name);
  }
  node_ptr->declare_parameter<double>(name, default_value, parameter_descriptor);
}

static Eigen::Index findMatching(const Eigen::VectorXd& key,
                                 const Eigen::MatrixXd& haystack,
                                 const Eigen::VectorXi& available) {
  Eigen::Index result = available.array().maxCoeff();
  double best_match = 0.0;
  for (unsigned int i = 0; i < available.rows(); ++i) {
    int index = available[i];
    if (index < 0)  // index already taken
      continue;
    if (index >= haystack.cols())
      return result;
    double match = haystack.col(available[i]).transpose() * key;
    double abs_match = std::abs(match);
    if (abs_match > 0.5 && abs_match > best_match) {
      best_match = abs_match;
      result = index;
    }
  }
  return result;
}
}  // namespace franka_example_controllers

// Expose the controller as visible to the rest of ros2_control
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerInterface)