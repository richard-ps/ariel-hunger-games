import mujoco
import numpy as np
from mujoco import viewer

import matplotlib.pyplot as plt
import wandb

import nevergrad as ng
from rich.progress import Progress

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
import ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_spider as hungry_spider
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
    return fig


def init(config):
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    if config['extra_bricks'] == True:
        if config['morphology'] == 'close':
            spider_core = hungry_spider.hungry_spider_close_z_plane_with_bricks()
        elif config['morphology'] == 'far':
            spider_core = hungry_spider.hungry_spider_far_z_plane_with_bricks()
        else:
            raise ValueError(f"'morphology' parameter had the value: {
                config['morphology']},\nwhich is not a valid parameter. Please choose a valid parameter of 'far' or 'close'!")
        world.spawn(spider_core.spec, position=[0, 0, 0])
    elif config['extra_bricks'] == False:
        if config['morphology'] == 'close':
            spider_core = hungry_spider.hungry_spider_close_z_plane()
            if config['movement'] == 'walk':
                world.spawn(spider_core.spec, position=[0, 0, 0.2])
            elif config['movement'] == 'crawl':
                world.spawn(spider_core.spec, position=[0, 0, 0])
            else:
                raise ValueError(f"'movement' parameter had the value: {
                    config['movement']},\nwhich is not a valid parameter. Please choose a valid parameter of 'walk' or 'crawl'!")

        elif config['morphology'] == 'far':
            spider_core = hungry_spider.hungry_spider_far_z_plane()
            if config['movement'] == 'walk':
                world.spawn(spider_core.spec, position=[0, 0, 0.1])
            elif config['movement'] == 'crawl':
                world.spawn(spider_core.spec, position=[0, 0, 0])
            else:
                raise ValueError(f"'movement' parameter had the value: {
                    config['movement']},\nwhich is not a valid parameter. Please choose a valid parameter of 'walk' or 'crawl'!")
        else:
            raise ValueError(f"'morphology' parameter had the value: {
                config['morphology']},\nwhich is not a valid parameter. Please choose a valid parameter of 'far' or 'close'!")
    else:
        raise ValueError(f"'extra_bricks' parameter had the value: {
                         config['extra_bricks']},\nwhich is not a valid parameter. Please choose a valid parameter of 'True' or 'False'!")

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
            0,
            2 * np.pi,
        ),

        w=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(0, 20),

        amplitudes=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            0,
            np.pi / 2,
        ),

        ha=ng.p.Array(shape=(len(data.ctrl),)).set_mutation(
            sigma=0.1).set_bounds(0, 1),

        b=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-20, 20),
    )

    optimizer = ng.optimizers.CMA(
        parametrization=params,
        budget=budget,
        num_workers=num_workers,
    )

    return optimizer


def log_data(run, idx, fitness, agent, plot, last_x_location):

    log_data = {}

    hinge_names = ['Right Leg Close', 'Right Leg Far', 'Left Leg Close', 'Left Leg Far',
                   'Front Leg Close', 'Front Leg Far', 'Back Leg Close', 'Back Leg Far']

    log_data['Agent Index'] = idx
    log_data["fitness"] = fitness
    log_data["Trajectory"] = plot
    log_data['Last X Location'] = last_x_location

    for key in agent.kwargs.keys():
        for hinge_name, value in zip(hinge_names, agent.kwargs[key]):
            log_data['Agent Index'] = idx
            log_data[f'{hinge_name} {key} Value'] = value

    run.log(log_data)
    plt.close(plot)


def main(run, fitness_func) -> None:
    """Entry function to run the simulation with random movements."""

    world, data, model, tracker, cpg_mat = init(run.config)

    optimizer = optimizer_setup(num_workers=run.config['num_workers'],
                                budget=run.config['budget'], optim=run.config['budget'], data=data)

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
    if run.config['extra_bricks'] == False:
        if run.config['movement'] == 'walk':
            if run.config['morphology'] == 'close':
                data.qpos[[7, 9, 11, 13]] = -np.pi / 2
            else:
                data.qpos[[8, 10, 12, 14]] = -np.pi / 2

    # Run optimization loop
    with Progress() as progress:
        task_agent_simulation = progress.add_task(
            "[red]Simulating Agents...", total=run.config['budget'])

        for idx in range(run.config['budget']):

            agent = optimizer.ask()

            ctrl.tracker.reset()
            if run.config['extra_bricks'] == False:
                if run.config['movement'] == 'walk':
                    if run.config['morphology'] == 'close':
                        data.qpos[[7, 9, 11, 13]] = -np.pi / 2
                    else:
                        data.qpos[[8, 10, 12, 14]] = -np.pi / 2

            cpg_mat.set_param_with_dict(agent.kwargs)
            simple_runner(
                model,
                data,
                duration=run.config['duration'],
                # steps_per_loop=run.config['steps'],
            )

            fitness = fitness_func(
                tracker.history["xpos"], run.config['penelty_multiplier'])

            # Need to make the fitness negative, since NeverGrad tries to minimize the loss/fitness function!
            optimizer.tell(agent, -fitness)

            progress.update(task_agent_simulation, advance=1)

            plot = show_xpos_history(tracker.history["xpos"][0], False)
            log_data(run, idx, fitness, agent,
                     plot, tracker.history["xpos"][0][-1][0])

    recommended_model = optimizer.recommend()
    recommended_params = recommended_model.kwargs
    print("BEST PARAMS: ", recommended_params)

    # Rerun best parameters
    cpg_mat.set_param_with_dict(recommended_params)
    ctrl.tracker.reset()
    mujoco.mj_resetData(model, data)
    if run.config['extra_bricks'] == False:
        if run.config['movement'] == 'walk':
            if run.config['morphology'] == 'close':
                data.qpos[[7, 9, 11, 13]] = -np.pi / 2
            else:
                data.qpos[[8, 10, 12, 14]] = -np.pi / 2

    simulate_best(model, data, tracker)

    run.finish(exit_code=0)


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

    # Define this before starting the evolution!
    def fitness_func(x_pos, penelty_multiplier):
        # Fitness Function for traveling in a streight line
        # return x_pos[0][-1][0] - penelty_multiplier * np.abs(x_pos[0][:][1] - 0.1).mean()
        # Fitness Function for traveling in a circle
        positions = np.column_stack((x_pos[0][:][0], x_pos[0][:][1] - 0.1))
        deltas = np.diff(positions, axis=0)
        headings = np.arctan2(deltas[:1], deltas[:0]).unwrap()
        total_radians = headings[-1] - headings[0]

        mean_radius = np.sqrt(
            (x_pos[0][:][0] ** 2 + (x_pos[0][:][1] - 0.1) ** 2)).mean()
        return total_radians - penelty_multiplier * mean_radius

    config = {
        # Agents per Generation
        # int
        'num_workers': 200,
        # Length of a Run
        # int
        'budget': 30_000,
        # Optimizer to use (from the NeverGrad Library)
        'optim': ng.optimizers.CMA,
        # User Defined fitness Function
        'fitness_func': fitness_func,
        # Penelty Multiplier (if applicable)
        # float
        'penelty_multiplier': 0,
        # Steps to take in a simulation
        # int
        'steps': 60,
        # Duration of a Simulation
        # int
        'duration': 60,
        # Body Morphology
        # close | far
        'morphology': 'close',
        # Movement Mode: Walk or crawl
        # crawl | walk
        'movement': 'crawl',
        # Sould it have bricks right after the first joints
        # True | False
        'extra_bricks': True,
    }

    if config["extra_bricks"]:
        project = "Hungry Gecko CMA-ES Evolution with Extra Bricks"
    elif not config["extra_bricks"]:
        project = "Hungry Gecko CMA-ES Evolution"
    else:
        raise ValueError(f"Config Parameter 'extra_bricks' has a value of {
                         config['extra_bricks']}, when only a value of True or False is allowed.")

    with wandb.init(project=project, config=config, mode="online") as run:
        main(run, config['fitness_func'])
