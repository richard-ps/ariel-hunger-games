import mujoco
import numpy as np
from mujoco import viewer

import matplotlib.pyplot as plt
import wandb

import nevergrad as ng
from rich.progress import Progress

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_gecko import hungry_gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner


def show_xpos_history(history: list[list[float]], show: bool):
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
    if show:
        plt.show()
    return ax


def init():
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = hungry_gecko()  # DO NOT CHANGE

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
    na_cpg_mat = NaCPG(create_fully_connected_adjacency(
        len(data.ctrl.copy())), angle_tracking=True)

    return world, data, model, tracker, na_cpg_mat


def optimizer_setup(num_workers, budget, optim, data) -> ng.optimizers:

    # Setup Nevergrad optimizer
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        w=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-4*np.pi, 8*np.pi),
        amplitudes=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        ha=ng.p.Array(shape=(len(data.ctrl),)).set_mutation(
            sigma=0.1).set_bounds(0, 1),
        b=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-100, 100),
    )

    optimizer = ng.optimizers.CMA(
        parametrization=params,
        budget=budget,
        num_workers=num_workers,
    )

    return optimizer


def main(config) -> None:
    """Entry function to run the simulation with random movements."""

    world, data, model, tracker, cpg_mat = init()

    optimizer = optimizer_setup(num_workers=config['num_workers'],
                                budget=config['budget'], optim=config['budget'], data=data)

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: cpg_mat.forward(d.time),
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
    with Progress() as progress:
        task_agent_simulation = progress.add_task(
            "[red]Simulating Agents...", total=config['budget'])

        for idx in range(config['budget']):

            agent = optimizer.ask()

            ctrl.tracker.reset()

            cpg_mat.set_param_with_dict(agent.kwargs)
            simple_runner(
                model,
                data,
                duration=config['duration'],
                steps_per_loop=config['steps'],
            )
            fitness = tracker.history["xpos"][0][-1][0]

            # Need to make the fitness negative, since NeverGrad tries to minimize the loss/fitness function!
            optimizer.tell(agent, fitness)

            progress.update(task_agent_simulation, advance=1)

            plot = show_xpos_history(tracker.history["xpos"][0], False)
            run.log({"Agent": idx,
                     "fitness": fitness,
                     "best_gen_agent_phase": agent.kwargs['phase'],
                     "best_gen_agent_w": agent.kwargs['w'],
                     "best_gen_agent_amplitudes": agent.kwargs['amplitudes'],
                     "best_gen_agent_ha": agent.kwargs['ha'],
                     "best_gen_agent_b": agent.kwargs['b'],
                     "Trajectory": plot,
                     })

    recommended_model = optimizer.recommend()
    recommended_params = recommended_model.kwargs
    print("BEST PARAMS: ", recommended_params)

    # Rerun best parameters
    cpg_mat.set_param_with_dict(recommended_params)
    ctrl.tracker.reset()
    mujoco.mj_resetData(model, data)

    simulate_best(model, data, tracker)


def simulate_best(model, data, tracker):
    # This opens a viewer window and runs the simulation with your controller
    viewer.launch(
        model=model,
        data=data,
    )

    show_xpos_history(tracker.history["xpos"][0], True)

    # Non-default VideoRecorder options
    PATH_TO_VIDEO_FOLDER = "./__videos__/"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)


# Render with video recorder
    video_renderer(
        model,
        data,
        duration=120,
        video_recorder=video_recorder,
    )


if __name__ == "__main__":
    wandb.login()

    project = "Hungry Gecko CMA-ES Evolution"

    config = {
        # Agents per Generation
        'num_workers': 200,
        # Length of a Run
        'budget': 10_000,
        # Optimizer to use (from the NeverGrad Library)
        'optim': ng.optimizers.CMA,
        # Steps to take in a simulation
        'steps': 100,
        # Duration of a Simulation
        'duration': 100,
    }
    with wandb.init(project=project, config=config) as run:
        main(config)
