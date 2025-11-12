from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from networkx import DiGraph
import networkx as nx
import matplotlib.pyplot as plt


def hungry_spider_far_z_plane():
    """
    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.

    Body Description
    ---------
    The gecko body consists of a core module, 4 legs (flippers), a neck and a spine.
    For better mobility the front two flippers have 2 hinges (joints), each rotated 90 degrees
    to each other. Additionally, the back two flippers are rotated 45 degrees compared to the
    bode, to encourage forward movement.
    """

    """
    Note. This configuration has the joints in the following rotations:
    Close Joints (close to the core): can move "left/right" (in the xy-plane).
    Far Joints (far away from the core): can move "up/down" (in the z-plane).

    Check out the configuration under this function if you need the opposite joint configuration.
    """

    core = CoreModule(
        index=0,
    )

    front_close_leg_hinge = HingeModule(
        index=1,
    )
    front_far_leg_hinge = HingeModule(
        index=2,
    )

    back_close_leg_hinge = HingeModule(
        index=3,
    )
    back_far_leg_hinge = HingeModule(
        index=4,
    )

    left_close_leg_hinge = HingeModule(
        index=5,
    )
    left_far_leg_hinge = HingeModule(
        index=6,
    )

    right_close_leg_hinge = HingeModule(
        index=7,
    )
    right_far_leg_hinge = HingeModule(
        index=8,
    )

    left_leg_brick = BrickModule(
        index=9,
    )
    right_leg_brick = BrickModule(
        index=10,
    )
    front_leg_brick = BrickModule(
        index=11,
    )
    back_leg_brick = BrickModule(
        index=12,
    )

    front_far_leg_hinge.rotate(90)
    back_far_leg_hinge.rotate(90)
    left_far_leg_hinge.rotate(90)
    right_far_leg_hinge.rotate(90)

    core.sites[ModuleFaces.BACK].attach_body(
        body=back_close_leg_hinge.body,
        prefix="back_close_leg_hinge",
    )
    back_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_far_leg_hinge.body,
        prefix="back_far_leg_hinge",
    )
    back_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_leg_brick.body,
        prefix="back_leg_brick",
    )

    core.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_hinge.body,
        prefix="front_close_leg_hinge",
    )
    front_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_far_leg_hinge.body,
        prefix="front_far_leg_hinge",
    )
    front_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_leg_brick.body,
        prefix="front_leg_brick",
    )

    core.sites[ModuleFaces.RIGHT].attach_body(
        body=right_close_leg_hinge.body,
        prefix="right_close_leg_hinge",
    )
    right_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_far_leg_hinge.body,
        prefix="right_far_leg_hinge",
    )
    right_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_leg_brick.body,
        prefix="right_leg_brick",
    )

    core.sites[ModuleFaces.LEFT].attach_body(
        body=left_close_leg_hinge.body,
        prefix="left_close_leg_hinge",
    )
    left_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_far_leg_hinge.body,
        prefix="left_far_leg_hinge",
    )
    left_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_leg_brick.body,
        prefix="left_leg_brick",
    )

    return core


def hungry_spider_close_z_plane():
    """
    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.

    Body Description
    ---------
    The gecko body consists of a core module, 4 legs (flippers), a neck and a spine.
    For better mobility the front two flippers have 2 hinges (joints), each rotated 90 degrees
    to each other. Additionally, the back two flippers are rotated 45 degrees compared to the
    bode, to encourage forward movement.
    """

    """
    Note. This configuration has the joints in the following rotations:
    Close Joints (close to the core): can move "up/down" (in the z-plane).
    Far Joints (far away from the core): can move "left/right" (in the xy-plane).

    Check out the configuration above this function if you need the opposite joint configuration.
    """

    core = CoreModule(
        index=0,
    )

    front_close_leg_hinge = HingeModule(
        index=1,
    )
    front_far_leg_hinge = HingeModule(
        index=2,
    )

    back_close_leg_hinge = HingeModule(
        index=3,
    )
    back_far_leg_hinge = HingeModule(
        index=4,
    )

    left_close_leg_hinge = HingeModule(
        index=5,
    )
    left_far_leg_hinge = HingeModule(
        index=6,
    )

    right_close_leg_hinge = HingeModule(
        index=7,
    )
    right_far_leg_hinge = HingeModule(
        index=8,
    )

    left_leg_brick = BrickModule(
        index=9,
    )
    right_leg_brick = BrickModule(
        index=10,
    )
    front_leg_brick = BrickModule(
        index=11,
    )
    back_leg_brick = BrickModule(
        index=12,
    )

    front_close_leg_hinge.rotate(90)
    back_close_leg_hinge.rotate(90)
    left_close_leg_hinge.rotate(90)
    right_close_leg_hinge.rotate(90)

    front_far_leg_hinge.rotate(90)
    back_far_leg_hinge.rotate(90)
    left_far_leg_hinge.rotate(90)
    right_far_leg_hinge.rotate(90)

    core.sites[ModuleFaces.BACK].attach_body(
        body=back_close_leg_hinge.body,
        prefix="back_close_leg_hinge",
    )
    back_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_far_leg_hinge.body,
        prefix="back_far_leg_hinge",
    )
    back_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_leg_brick.body,
        prefix="back_leg_brick",
    )

    core.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_hinge.body,
        prefix="front_close_leg_hinge",
    )
    front_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_far_leg_hinge.body,
        prefix="front_far_leg_hinge",
    )
    front_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_leg_brick.body,
        prefix="front_leg_brick",
    )

    core.sites[ModuleFaces.RIGHT].attach_body(
        body=right_close_leg_hinge.body,
        prefix="right_close_leg_hinge",
    )
    right_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_far_leg_hinge.body,
        prefix="right_far_leg_hinge",
    )
    right_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_leg_brick.body,
        prefix="right_leg_brick",
    )

    core.sites[ModuleFaces.LEFT].attach_body(
        body=left_close_leg_hinge.body,
        prefix="left_close_leg_hinge",
    )
    left_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_far_leg_hinge.body,
        prefix="left_far_leg_hinge",
    )
    left_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_leg_brick.body,
        prefix="left_leg_brick",
    )

    return core


def hungry_spider_close_z_plane_with_bricks():
    """
    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.

    Body Description
    ---------
    The gecko body consists of a core module, 4 legs (flippers), a neck and a spine.
    For better mobility the front two flippers have 2 hinges (joints), each rotated 90 degrees
    to each other. Additionally, the back two flippers are rotated 45 degrees compared to the
    bode, to encourage forward movement.
    """

    """
    Note. This configuration has the joints in the following rotations:
    Close Joints (close to the core): can move "up/down" (in the z-plane).
    Far Joints (far away from the core): can move "left/right" (in the xy-plane).

    Check out the configuration above this function if you need the opposite joint configuration.
    """

    core = CoreModule(
        index=0,
    )

    front_close_leg_hinge = HingeModule(
        index=1,
    )
    front_far_leg_hinge = HingeModule(
        index=2,
    )

    back_close_leg_hinge = HingeModule(
        index=3,
    )
    back_far_leg_hinge = HingeModule(
        index=4,
    )

    left_close_leg_hinge = HingeModule(
        index=5,
    )
    left_far_leg_hinge = HingeModule(
        index=6,
    )

    right_close_leg_hinge = HingeModule(
        index=7,
    )
    right_far_leg_hinge = HingeModule(
        index=8,
    )

    left_close_leg_brick = BrickModule(
        index=9,
    )
    right_close_leg_brick = BrickModule(
        index=10,
    )
    front_close_leg_brick = BrickModule(
        index=11,
    )
    back_close_leg_brick = BrickModule(
        index=12,
    )
    left_far_leg_brick = BrickModule(
        index=13,
    )
    right_far_leg_brick = BrickModule(
        index=14,
    )
    front_far_leg_brick = BrickModule(
        index=15,
    )
    back_far_leg_brick = BrickModule(
        index=16,
    )

    front_close_leg_hinge.rotate(90)
    back_close_leg_hinge.rotate(90)
    left_close_leg_hinge.rotate(90)
    right_close_leg_hinge.rotate(90)

    front_far_leg_hinge.rotate(90)
    back_far_leg_hinge.rotate(90)
    left_far_leg_hinge.rotate(90)
    right_far_leg_hinge.rotate(90)

    core.sites[ModuleFaces.BACK].attach_body(
        body=back_close_leg_hinge.body,
        prefix="back_close_leg_hinge",
    )
    back_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_close_leg_brick.body,
        prefix="back_close_leg_brick",
    )
    back_close_leg_brick.sites[ModuleFaces.FRONT].attach_body(
        body=back_far_leg_hinge.body,
        prefix="back_far_leg_hinge",
    )
    back_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_far_leg_brick.body,
        prefix="back_leg_brick",
    )

    core.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_hinge.body,
        prefix="front_close_leg_hinge",
    )
    front_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_brick.body,
        prefix="front_close_leg_brick",
    )
    front_close_leg_brick.sites[ModuleFaces.FRONT].attach_body(
        body=front_far_leg_hinge.body,
        prefix="front_far_leg_hinge",
    )
    front_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_far_leg_brick.body,
        prefix="front_leg_brick",
    )

    core.sites[ModuleFaces.RIGHT].attach_body(
        body=right_close_leg_hinge.body,
        prefix="right_close_leg_hinge",
    )
    right_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_close_leg_brick.body,
        prefix="right_close_leg_brick",
    )
    right_close_leg_brick.sites[ModuleFaces.FRONT].attach_body(
        body=right_far_leg_hinge.body,
        prefix="right_far_leg_hinge",
    )
    right_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_far_leg_brick.body,
        prefix="right_leg_brick",
    )

    core.sites[ModuleFaces.LEFT].attach_body(
        body=left_close_leg_hinge.body,
        prefix="left_close_leg_hinge",
    )
    left_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_close_leg_brick.body,
        prefix="left_close_leg_brick",
    )
    left_close_leg_brick.sites[ModuleFaces.FRONT].attach_body(
        body=left_far_leg_hinge.body,
        prefix="left_far_leg_hinge",
    )
    left_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_far_leg_brick.body,
        prefix="left_leg_brick",
    )

    return core


def hungry_spider_far_z_plane_with_bricks():
    """
    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.

    Body Description
    ---------
    The gecko body consists of a core module, 4 legs (flippers), a neck and a spine.
    For better mobility the front two flippers have 2 hinges (joints), each rotated 90 degrees
    to each other. Additionally, the back two flippers are rotated 45 degrees compared to the
    bode, to encourage forward movement.
    """

    """
    Note. This configuration has the joints in the following rotations:
    Close Joints (close to the core): can move "up/down" (in the z-plane).
    Far Joints (far away from the core): can move "left/right" (in the xy-plane).

    Check out the configuration above this function if you need the opposite joint configuration.
    """

    core = CoreModule(
        index=0,
    )

    front_close_leg_hinge = HingeModule(
        index=1,
    )
    front_far_leg_hinge = HingeModule(
        index=2,
    )

    back_close_leg_hinge = HingeModule(
        index=3,
    )
    back_far_leg_hinge = HingeModule(
        index=4,
    )

    left_close_leg_hinge = HingeModule(
        index=5,
    )
    left_far_leg_hinge = HingeModule(
        index=6,
    )

    right_close_leg_hinge = HingeModule(
        index=7,
    )
    right_far_leg_hinge = HingeModule(
        index=8,
    )

    left_close_leg_brick = BrickModule(
        index=9,
    )
    right_close_leg_brick = BrickModule(
        index=10,
    )
    front_close_leg_brick = BrickModule(
        index=11,
    )
    back_close_leg_brick = BrickModule(
        index=12,
    )
    left_far_leg_brick = BrickModule(
        index=13,
    )
    right_far_leg_brick = BrickModule(
        index=14,
    )
    front_far_leg_brick = BrickModule(
        index=15,
    )
    back_far_leg_brick = BrickModule(
        index=16,
    )

    front_far_leg_hinge.rotate(90)
    back_far_leg_hinge.rotate(90)
    left_far_leg_hinge.rotate(90)
    right_far_leg_hinge.rotate(90)

    core.sites[ModuleFaces.BACK].attach_body(
        body=back_close_leg_hinge.body,
        prefix="back_close_leg_hinge",
    )
    back_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_close_leg_brick.body,
        prefix="back_close_leg_brick",
    )
    back_close_leg_brick.sites[ModuleFaces.BOTTOM].attach_body(
        body=back_far_leg_hinge.body,
        prefix="back_far_leg_hinge",
    )
    back_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=back_far_leg_brick.body,
        prefix="back_leg_brick",
    )

    core.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_hinge.body,
        prefix="front_close_leg_hinge",
    )
    front_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_close_leg_brick.body,
        prefix="front_close_leg_brick",
    )
    front_close_leg_brick.sites[ModuleFaces.BOTTOM].attach_body(
        body=front_far_leg_hinge.body,
        prefix="front_far_leg_hinge",
    )
    front_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=front_far_leg_brick.body,
        prefix="front_leg_brick",
    )

    core.sites[ModuleFaces.RIGHT].attach_body(
        body=right_close_leg_hinge.body,
        prefix="right_close_leg_hinge",
    )
    right_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_close_leg_brick.body,
        prefix="right_close_leg_brick",
    )
    right_close_leg_brick.sites[ModuleFaces.BOTTOM].attach_body(
        body=right_far_leg_hinge.body,
        prefix="right_far_leg_hinge",
    )
    right_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=right_far_leg_brick.body,
        prefix="right_leg_brick",
    )

    core.sites[ModuleFaces.LEFT].attach_body(
        body=left_close_leg_hinge.body,
        prefix="left_close_leg_hinge",
    )
    left_close_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_close_leg_brick.body,
        prefix="left_close_leg_brick",
    )
    left_close_leg_brick.sites[ModuleFaces.BOTTOM].attach_body(
        body=left_far_leg_hinge.body,
        prefix="left_far_leg_hinge",
    )
    left_far_leg_hinge.sites[ModuleFaces.FRONT].attach_body(
        body=left_far_leg_brick.body,
        prefix="left_leg_brick",
    )

    return core
