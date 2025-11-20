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


def distance_to_target(
    initial_position: tuple[float, float, float],
    target_position: tuple[float, float, float],   
    
) -> float:
    p = [initial_position[0],initial_position[1]]
    q = [target_position[0], target_position[1]]

    # Calculate Euclidean distance
    return math.dist(p, q)
    # return  (target_position[0] - initial_position[0]) + -0.8 *(target_position[1] - abs(initial_position[1]))

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

    # Setup Nevergrad optimizer
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        w=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-4*np.pi, 8*np.pi),
        amplitudes=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -np.pi/2,
            np.pi/2,
        ),
        ha=ng.p.Array(shape=(len(data.ctrl),)).set_mutation(sigma=0.01).set_bounds(0, 1),
        b=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-100, 100),
    )
    num_of_workers = 100
    budget = 200
    optim = ng.optimizers.PSO
    optimizer = optim(
        parametrization=params,
        budget=budget,
        num_workers=num_of_workers,
    )

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

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Run optimization loop
    best_fitness = 0.0
    best_params = None
    for idx in range(optimizer.budget):
        ctrl.tracker.reset()
        x = optimizer.ask()
        # print("X: ", x)
        na_cpg_mat.set_param_with_dict(x.kwargs)
        # angles = na_cpg_mat.forward()
        # print("Angles: ", angles)
        simple_runner(
            model,
            data,
            duration=10,
            steps_per_loop=100,
        )
        # print("TRACKER HISTORY XPOS: ", tracker.history["xpos"][0][-1])
        # distance = -1 * tracker.history["xpos"][0][-1][0]
        geom_id = data.geom('robot1_core').id
        rotation_matrix = data.geom_xmat[geom_id].reshape(3, 3)
        # quat = data.body(body_id).axisangle
        print("Rotation: ", rotation_matrix)
        # Create a Rotation object from the 3x3 matrix
        rot = R_scipy.from_matrix(rotation_matrix)

        # Convert to ZYX Euler angles (Yaw, Pitch, Roll)
        # The output order is (Yaw, Pitch, Roll) based on the 'zyx' sequence
        # euler_angles_rad = rot.as_euler('zyx', degrees=False)
        euler_angles_deg = rot.as_euler('zyx', degrees=True)

        # print(f"\nEuler Angles (ZYX sequence):")
        # print(f"  [Yaw (Z), Pitch (Y), Roll (X)] in radians: {euler_angles_rad}")
        # print(f"  [Yaw (Z), Pitch (Y), Roll (X)] in degrees: {euler_angles_deg}")
        distance = abs(euler_angles_deg[1]) - 0.5 * abs(distance_to_target(
            tracker.history["xpos"][0][-1],
            [0.0, 0.0, 0.0],
        ))
        # if distance < -40.0:
        #     console.log("Target reached!")
        #     best_fitness = distance
        #     best_params = x.kwargs
        #     console.log(
        #         f"({idx}) Current distance: {-1*distance}, Best distance: {-1*best_fitness}",
        #     )
        #     break
        print("IDX", idx, "DISTANCE: ", -1*distance )
        optimizer.tell(x, distance)
        if abs(distance) > abs(best_fitness):
            best_fitness = distance
            best_params = x.kwargs
            console.log(
                f"({idx}) Current distance: {-1*distance}, Best distance: {-1*best_fitness}",
            )

    print("BEST PARAMS: ", best_params)

    # Rerun best parameters
    na_cpg_mat.set_param_with_dict(best_params)
    ctrl.tracker.reset()
    mujoco.mj_resetData(model, data)

    # This opens a viewer window and runs the simulation with your controller
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,
        data=data,
    )

    show_xpos_history(tracker.history["xpos"][0])

    # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=120,
    #     video_recorder=video_recorder,
    # )

def print_cpg():
    # WALK
    # parameters = {'phase': [ 1.84770147,  0.63517527, -6.2581397 ,  5.79072771,  3.92560091,
    # -3.12365597, -5.64161065,  1.61769557], 'w': [ -3.82029392, -10.09482901,   7.54136557,  -9.10054186,
    #     -9.74108929,  16.49676456,  22.68651804,  12.8404796 ], 'amplitudes': [ 0.41219116,  4.24333571, -4.81600483, -2.72279552,  3.86351227,
    #     4.13802726, -5.23314684, -4.10500115], 'ha': [0.73259639, 0.84562198, 0.60696611, 0.74627439, 0.26942003,
    # 0.16558826, 0.04662044, 0.05292187], 'b': [-20.68912344,  59.14947598,  50.71959856,  15.96840235,
    # -88.9128633 ,  33.98478395, -33.86570571,  85.19837797]}

    # ROTATE
    parameters = {'phase': [ 0.39893007, -2.23096376, -5.4295317 , -1.57314901,  2.55956377,
       -2.14105615, -1.04753494, -3.77477379], 'w': [ -0.94608458,  17.24350967,  22.77230644,   9.2783865 ,
       -10.5069646 ,  10.12239204,  10.10922897,  18.80313651], 'amplitudes': [-0.96266415, -1.077364  , -1.41101981, -1.0691907 ,  0.83289887,
       -1.27151085, -0.58288134, -1.20677418], 'ha': [0.04670639, 0.13176528, 0.7126964 , 0.832918  , 0.19586868,
       0.85530159, 0.82514118, 0.03681464], 'b': [-37.11845883,  91.33867811,  74.86399711, -33.95453279,
        11.02680772, -11.96790805, -57.62349316, -21.65943134]}

    # rotate parameters for testing
    
    adj_dict = create_fully_connected_adjacency(8)
    na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)
    na_cpg_mat.set_param_with_dict(parameters)
    for _ in range(800):
        na_cpg_mat.forward()

    import matplotlib.pyplot as plt

    hist = torch.tensor(na_cpg_mat.angle_history)
    times = torch.arange(hist.shape[0]) * na_cpg_mat.dt

    plt.figure(figsize=(8, 4))
    for j in range(hist.shape[1]):
        plt.plot(times, hist[:, j], label=f"joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("angle")
    plt.title("CPG angle histories")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(DATA / "angle_histories.png")
    plt.show()

if __name__ == "__main__":
    main()
    # print_cpg()
