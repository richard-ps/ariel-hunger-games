import mujoco
from mujoco import viewer

import numpy

from ariel.simulation.controllers.controller import Controller, Tracker
import ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_spider as hungry_spider
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner


def init(robot_model):

    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    robot_core = robot_model()  # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot_core.spec, position=[0, 0, 0.2])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return world, data, model


def simulate_best(model, data, tracker):
    # This opens a viewer window and runs the simulation with your controller
    # data.qpos[[8, 10, 12, 14]] = -numpy.pi / 2
    # data.qpos[2] = 0.175
    data.qpos[[7, 9, 11, 13]] = -numpy.pi / 2
    # data.qpos[2] = 0.275
    viewer.launch(
        model=model,
        data=data,
    )


def main():
    world, data, model = init(
        hungry_spider.hungry_spider_far_z_plane_with_bricks)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    simulate_best(model, data, None)


if __name__ == "__main__":
    main()
