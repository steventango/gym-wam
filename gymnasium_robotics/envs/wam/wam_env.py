from typing import Union

import numpy as np

from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from gymnasium import spaces

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": 0,
    "lookat": np.array([0.47, 0, 0.69]),
}


def get_base_wam_env(RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]):
    """Factory function that returns a BaseWAMEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseWAMEnv(RobotEnvClass):
        """Superclass for all WAM environments."""

        def __init__(
            self,
            gripper_extra_height,
            block_gripper,
            has_object: bool,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            reward_type,
            dof,
            **kwargs
        ):
            """Initializes a new WAM environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
                dof (int): the number of degrees of freedom of the robot
            """

            self.gripper_extra_height = gripper_extra_height
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.target_in_the_air = target_in_the_air
            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type
            self.dof = dof
            self.C = np.zeros((2, 3, 4))
            self.C[:] = np.eye(3, 4)
            self.width = 640
            self.height = 480
            self.goal3d = np.zeros(3)
            if self.dof == 3:
                self.active_joint_indices = [0, 1, 3]
                self.observation_joint_indices = self.active_joint_indices + [4]
            else:
                self.active_joint_indices = list(range(self.dof))
                self.observation_joint_indices = self.active_joint_indices
            super().__init__(n_actions=dof, **kwargs)
            self.action_space = spaces.Box(-0.7, 0.7, shape=(dof,), dtype="float32")

        # GoalEnv methods
        # ----------------------------
        def goal_distance(self, goal_a, goal_b, dim=3):
            assert goal_a.shape == goal_b.shape
            if dim == 3:
                goal_a = self.achieved_goal
                goal_b = self.goal3d
                return np.linalg.norm(goal_a - goal_b, axis=-1)
            goal_a = goal_a.ravel()
            goal_b = goal_b.ravel()
            return np.linalg.norm(goal_a - goal_b, axis=-1)

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            if self.reward_type == "sparse":
                return self._is_success(achieved_goal, goal)
            else:
                d = self.goal_distance(achieved_goal, goal, dim=2)
                return -d

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (self.dof,)
            return action

        def _get_obs(self):
            (
                grip_pos,
                object_pos,
                object_rel_pos,
                gripper_state,
                object_rot,
                object_velp,
                object_velr,
                grip_velp,
                gripper_vel,
                target_pos,
                robot_qpos,
                robot_qvel,
            ) = self.generate_mujoco_observations()

            if not self.has_object:
                achieved_goal = self._project_pos(grip_pos).ravel()
            else:
                achieved_goal = np.squeeze(object_pos.copy())

            obs = np.concatenate(
                [
                    robot_qpos,
                    robot_qvel
                ]
            )

            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }

        def _project_pos(self, pos):
            min_shape = np.min([self.width, self.height])
            pos_h = np.ones((4,))
            pos_h[:3] = pos
            img = self.C @ pos_h
            img = img[:, :2] / img[:, 2]
            img[:, 0] -= self.width / 2
            img[:, 1] -= self.height / 2
            img /= min_shape
            return img

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
            else:
                goal = self.np_random.uniform(
                    [0.25, -0.35, 0.5],
                    [0.75, 0.35, 0.8]
                )
            self.goal3d = goal.copy()
            goal_img = self._project_pos(goal).ravel()
            return goal_img

        def _is_success(self, achieved_goal, desired_goal):
            d = self.goal_distance(achieved_goal, desired_goal, dim=3)
            return (d < self.distance_threshold).astype(np.float32)

        def compute_terminated(self, achieved_goal, desired_goal, info):
            """Terminate the episode if the goal is achieved."""
            return bool(self._is_success(achieved_goal, desired_goal))

    return BaseWAMEnv


class MujocoPyWAMEnv(get_base_wam_env(MujocoPyRobotEnv)):
    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self.sim.data.qvel[self.active_joint_indices] = action

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt

        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
        robot_qpos = robot_qpos[self.observation_joint_indices]
        robot_qvel = robot_qvel[self.observation_joint_indices]

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            # velocities
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velr = self.sim.data.get_site_xvelr("object0") * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        target_pos = self.sim.data.get_site_xpos("target0")

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
            target_pos
        )

    def _get_gripper_xpos(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        return self.sim.data.body_xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal3d - sites_offset[0]
        self.sim.forward()

    def _viewer_setup(self):
        lookat = self._get_gripper_xpos()
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]


class MujocoWAMEnv(get_base_wam_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self.data.qvel[self.active_joint_indices] = action

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        robot_qpos = robot_qpos[self.observation_joint_indices]
        robot_qvel = robot_qvel[self.observation_joint_indices]
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        target_pos = self.goal3d - sites_offset[0]

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
            target_pos,
            robot_qpos,
            robot_qvel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:grip"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal3d - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        # Randomize positions of cameras
        self._reset_cams()

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]

    def _reset_cams(self):
        azimuth_eps = 150
        distance_eps = 0.5
        elevation_eps = 5
        orthogonal_eps = 5
        randomize = False

        # randomize cam0 position
        cam0_config = DEFAULT_CAMERA_CONFIG.copy()
        if randomize:
            cam0_config["azimuth"] += self.np_random.uniform(-azimuth_eps, azimuth_eps)
            cam0_config["distance"] += self.np_random.uniform(-distance_eps, distance_eps)
            cam0_config["elevation"] += self.np_random.uniform(-elevation_eps, elevation_eps)
        else:
            cam0_config["azimuth"] = -135
        self._cam_setup(0, cam0_config)

        # randomize cam1 position
        cam_1_config = DEFAULT_CAMERA_CONFIG.copy()
        if randomize:
            if self.np_random.uniform() < 0.5:
                cam_1_config["azimuth"] = cam0_config["azimuth"] + 90
            else:
                cam_1_config["azimuth"] = cam0_config["azimuth"] - 90
            cam_1_config["azimuth"] += self.np_random.uniform(-orthogonal_eps, orthogonal_eps)
            cam_1_config["distance"] += self.np_random.uniform(-distance_eps, distance_eps)
            cam_1_config["elevation"] += self.np_random.uniform(-elevation_eps, elevation_eps)
        else:
            cam_1_config["azimuth"] = cam0_config["azimuth"] - 90
        self._cam_setup(1, cam_1_config)

    def _cam_setup(self, id, config):
        cam = self.model.cam(id)
        lookat = config["lookat"]
        distance = config["distance"]
        azimuth = np.radians(config["azimuth"])
        elevation = np.radians(config["elevation"])
        pos = lookat.copy()
        pos[0] += distance * np.cos(elevation) * np.cos(azimuth)
        pos[1] += distance * np.cos(elevation) * np.sin(azimuth)
        pos[2] += distance * np.sin(elevation)

        cam.pos[:] = pos

        euler = [np.pi / 2 + elevation, np.pi/2 + azimuth, 0]

        cam.quat[:] = rotations.euler2quat(euler)

        fov = self.model.vis.global_.fovy

        # https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L736
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rotations.euler2mat(euler).T
        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0

        self.C[id] = image @ focal @ rotation @ translation
