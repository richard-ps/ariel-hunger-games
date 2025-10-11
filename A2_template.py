# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import os
import neat
# import visualize


# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.utils.random_morphology import new_robot
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.simulation.controllers.hopfs_cpg import HopfCPG

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.hungry_gecko import hungry_gecko

import time

# Keep track of data / history
HISTORY = []

class MujocoWrapper():
    def __init__(self, model, data, core, mujoco):
        self.model = model
        self.data = data
        self.core = core
        self.mujoco = mujoco

    def eval_genomes(self, genomes, config):
        """Evaluate genomes in the context of the simulation."""
        # Small movement

    

        for id, genome in genomes:
            print("Genome ID:", genome)
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            adjacency_list = {}

            for i in range(8):
                adjacency_list[i] = [(i + 1) % 8, (i - 1) % 8]  # Ring topology

            parameters = net.activate((self.data.ctrl))

            amplitudes = parameters[0:8]
            alpha = parameters[8:16]
            omega = parameters[16:24]   
            phase_diff = parameters[24]
            h = parameters[25]
            
            # List to hold CPG instances
            cpgs = []

            for i in range(8):
                cpg = HopfCPG(A=amplitudes,
                        num_neurons=8, 
                        adjacency_list=adjacency_list,
                        dt=0.02, 
                        h=h, 
                        alpha=alpha,
                        omega=omega,
                        phase_diff=phase_diff 
                        )
                cpgs.append(cpg)

                # x, y = cpgs[i].simulate(1500)

                # plt.plot(y*cpgs[i].A, label=f'CPG')
                # plt.title('CPG Outputs Over Time')
                # plt.xlabel('Time Steps')
                # plt.ylabel('Output')
                # plt.legend()
                # plt.show()
            
            for i in range(500):
                moves = []
                for cpg in range(len(cpgs)):
                    _, y = cpgs[cpg].step()
                    moves.append(y[cpg] * cpgs[cpg].A[cpg])

                # print("Moves Before:", moves)
                moves = np.array(moves) * 90.0
                # print("Moves:", moves)
                self.data.ctrl = np.clip(moves, -np.pi/2, np.pi/2)
                self.mujoco.mj_step(self.model, self.data)  # Step the simulation

            # print("Final Position:", self.core.xpos)
            
            #genome.fitness = distance_to_target((self.core.xpos[0], self.core.xpos[1]), (1.0, 0.0))
            genome.fitness = self.core.xpos[0].copy()
            print("Genome Fitness:", genome.fitness)
            mujoco.mj_resetData(self.model, self.data)  # Reset simulation data for the next genome

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    #new_robot_body = new_robot()
    new_robot_body = hungry_gecko()
    world.spawn(new_robot_body.spec, spawn_position=[0, 0, 0])
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    wrapper = MujocoWrapper(model, data, to_track[0], mujoco)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    # p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    # winner = p.run(wrapper.eval_genomes, 100)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    def brain_controller(model, data, to_track):
        # move = winner_net.activate((to_track[0].xpos[0], to_track[0].xpos[1]))
        parameters = winner_net.activate((data.ctrl))
        
        amplitudes = parameters[0:8]
        alpha = parameters[8:16]
        omega = parameters[16:24]   
        phase_diff = parameters[24]
        h = parameters[25]

        adjacency_list = {}

        for i in range(8):
            adjacency_list[i] = [(i + 1) % 8, (i - 1) % 8]  # Ring topology

        # List to hold CPG instances
        cpgs = []

        for i in range(8):
            cpg = HopfCPG(A=[45]*8,
                    num_neurons=8, 
                    adjacency_list=adjacency_list,
                    dt=0.02, 
                    h=h, 
                    alpha=alpha,
                    omega=omega,
                    phase_diff=phase_diff 
                    )
            cpgs.append(cpg)

        moves = []
        for cpg in range(len(cpgs)):
            _, y = cpgs[cpg].step()
            moves.append(y[cpg])

        # print("Moves Before:", moves)
        # moves = np.array(moves) * 90.0
        # print("Moves:", moves)
        moves = np.array(moves) * 0.05
        clipped_moves = np.clip(moves, -90, 90)
        # print("Clipped Moves:", clipped_moves)
        data.ctrl = clipped_moves

        HISTORY.append(to_track[0].xpos.copy())
        # print("Move Brain:", move)

    # Set the control callback function
    # This is called every time step to get the next action. 
    mujoco.set_mjcb_control(lambda m,d: brain_controller(m,d,to_track))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()


