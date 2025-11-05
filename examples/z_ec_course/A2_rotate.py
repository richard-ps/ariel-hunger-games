"""A2: Template for creating a custom controller using NaCPG."""

# Standard libraries
import math
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
import torch
from mujoco import viewer
from scipy.spatial.transform import Rotation as R_scipy

# import prebuilt robot phenotypes
from ariel import console
# from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_gecko import hungry_gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [0.0, 0.0, 0.0]
# TARGET_POSITION = [5.0, 0.0, 0.0]


# def distance_to_target(
#     initial_position: tuple[float, float, float],
#     target_position: tuple[float, float, float],   
    
# ) -> float:
#     p = [initial_position[0],initial_position[1]]
#     q = [target_position[0], target_position[1]]

#     # Calculate Euclidean distance
#     return math.dist(p, q)
#     # return  (target_position[0] - initial_position[0]) + -0.8 *(target_position[1] - abs(initial_position[1]))

def show_xpos_history(history: list[list[float]]) -> None:
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    plt.show()


def main() -> None:
    """Entry function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = hungry_gecko()  # DO NOT CHANGE
    # gecko_core.spec.frame = mujoco.MjFrame()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, position=[0, 0, 0])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Setup the NaCPG controller
    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)
    na_cpg_mat.set_param_with_dict({'phase': [ 6.28318531,  6.28318531, -6.28318531,  6.28318531,  0.44291052,
        6.28318531,  6.28318531,  0.48849384], 'w': [-12.56637061, -12.56637061,   0.66234324,  -0.38651667,
        -0.03509073,   0.35701786,  -1.70384422,  -0.16974864], 'amplitudes': [-1.047039  ,  0.98827418,  1.57079633, -1.57079633, -1.57079633,
        0.8845958 , -0.78082449, -0.59788558], 'ha': [0.        , 0.        , 0.00658146, 0.00558966, 0.00711154,
       0.00676326, 1.        , 0.00204783], 'b': [-5.61352178e-01,  2.45738617e-01,  6.12423700e-02,  1.00000000e+02,
        8.99765969e-01,  1.00000000e+02,  1.00000000e+02,  1.00000000e+02]})

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
        tracker=tracker,
    )

    # Set the control callback function
    # This is called every time step to get the next action.
    # Pass the model and data to the tracker
    ctrl.tracker.setup(world.spec, data)

    # Set the control callback function
    mujoco.set_mjcb_control(
        ctrl.set_control,
    )

    ctrl.tracker.reset()

    # This opens a viewer window and runs the simulation with your controller
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,
        data=data,
    )

    show_xpos_history(tracker.history["xpos"][0])

    # Non-default VideoRecorder options
    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # Render with video recorder
    video_renderer(
        model,
        data,
        duration=120,
        video_recorder=video_recorder,
    )

if __name__ == "__main__":
    main()
    # print_cpg()
