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
        'amplitudes': [1.54995063, 0.74874286, 1.34209176, 1.34221212, 1.0869121, 0.75970761, 1.27984321, 1.40854025],
        'phase': [1.78395141, 3.01059952, 4.91756051, 3.03321597, 6.28318531, 1.18410547, 1.23806811, 2.30030358],
        'ha': [0.74712867, 0.31810203, 0.2558637, 0.83425911, 0.48746279, 0.03894261, 0.56317329, 0.00588796],
        'b': [0.08136917, -2.18100014,  0.60056972, -0.2587887,  7.32292553, -0.80629822,  2.14765765,  4.15820492],
        'w': [11.85802214,  8.36197005, 12.28270002, 14.83555679,  7.97016965, 9.30251057, 13.42179372, 11.14953905],
    }
    main(config)
