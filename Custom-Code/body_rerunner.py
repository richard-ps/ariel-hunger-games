"""A2: Template for creating a custom controller using NaCPG."""

# Standard libraries
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import viewer

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
import ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_spider as hungry_spider
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld

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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot x,y trajectory
    ax.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    ax.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    ax.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_title("Robot Path in XY Plane")
    ax.legend()
    ax.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    ax.axis("equal")
    plt.show()

    return ax


def main(config: dict) -> None:
    """Entry function to run the simulation with random movements."""
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    world = SimpleFlatWorld()

    gecko_core = hungry_spider.hungry_spider_far_z_plane()

    world.spawn(gecko_core.spec, position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)
    na_cpg_mat.set_param_with_dict({
        'phase': config['phase'],
        'w': config['w'],
        'amplitudes': config['amplitudes'],
        'ha': config['ha'],
        'b': config['b'],
    })

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
        tracker=tracker,
    )

    ctrl.tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(
        ctrl.set_control,
    )

    ctrl.tracker.reset()

    viewer.launch(
        model=model,
        data=data,
    )

    show_xpos_history(tracker.history["xpos"][0])

    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    video_renderer(
        model,
        data,
        duration=120,
        video_recorder=video_recorder,
    )


if __name__ == "__main__":
    config = {
        'amplitudes': [1.50243896, 1.56258453, 1.29187972, 1.56163921, 1.4741377, 1.367098, 1.41342346, 0.5566824],
        'phase': [3.33081558, 4.41174703, 2.68773247, 0.84130775, 5.59803051, 0.08632286, 1.28354494, 3.94187516],
        'w': [9.10791933,  7.22536897,  8.90013522,  5.89307335,  6.66477066, 12.03859487,  9.52397259, 10.07842654],
        'ha': [0.1574605, 0.53079165, 0.54345309, 0.34459765, 0.36877258, 0.12627963, 0.21800283, 0.60473463],
        'b': [-2.26778477, -0.04779329, -1.42405013,  0.99285877, -1.29151324, 3.81529111, -2.9382482,  0.76793178],
    }
    main(config)
