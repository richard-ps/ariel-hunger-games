from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
import numpy as np

def hungry_gecko():
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

    core = CoreModule(
        index=0,
    )
    
    fl_leg = HingeModule(
        index=5,
    )
    fl_leg.rotate(180)
    fl_leg2 = HingeModule(
        index=15,
    )
    fl_leg2.rotate(90)
    
    fl_flipper = BrickModule(
        index=6,
    )
    
    fr_leg = HingeModule(
        index=7,
    )
    fr_leg.rotate(180)
    fr_leg2 = HingeModule(
        index=17,
    )
    fr_leg2.rotate(90)
    
    fr_flipper = BrickModule(
        index=8,
    )
    bl_leg = HingeModule(
        index=9,
    )
    bl_leg.rotate(180)
    bl_flipper = BrickModule(
        index=10,
    )
    br_leg = HingeModule(
        index=11,
    )
    br_leg.rotate(90)
    br_flipper = BrickModule(
        index=12,
    )

    fl2_leg = HingeModule(
        index=13,
    )
    fl2_leg.rotate(180)
    fl2_leg2 = HingeModule(
        index=14,
    )
    fl2_leg2.rotate(90)

    br2_leg = BrickModule(
        index=16,
    )

    fl3_leg = HingeModule(
        index=18,
    )
    fl3_leg.rotate(180)
    fl3_leg3 = HingeModule(
        index=19,
    )
    fl3_leg3.rotate(90)

    br3_leg = BrickModule(
        index=20,
    )

    core.sites[ModuleFaces.BACK].attach_body(
        body=fl2_leg.body,
        prefix="fl2_leg",
    )

    fl2_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fl2_leg2.body,
        prefix="fl2_flipper",
    )

    fl2_leg2.sites[ModuleFaces.FRONT].attach_body(
        body=br2_leg.body,
        prefix="br2_flipper",
    )

    core.sites[ModuleFaces.FRONT].attach_body(
        body=fl3_leg.body,
        prefix="fl3_leg",
    )

    fl3_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fl3_leg3.body,
        prefix="fl3_flipper",
    )

    fl3_leg3.sites[ModuleFaces.FRONT].attach_body(
        body=br3_leg.body,
        prefix="br3_flipper",
    )

    core.sites[ModuleFaces.LEFT].attach_body(
        body=fl_leg.body,
        prefix="fl_leg",
    )
    
    fl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fl_leg2.body,
        prefix="fl_flipper",
    )
    fl_leg2.sites[ModuleFaces.FRONT].attach_body(
        body=fl_flipper.body,
        prefix="fl_flipper2",
    )
    
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=fr_leg.body,
        prefix="fr_leg",
    )
    
    fr_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fr_leg2.body,
        prefix="fr_flipper",
    )
    fr_leg2.sites[ModuleFaces.FRONT].attach_body(
        body=fr_flipper.body,
        prefix="fr_flipper2",
    )
    
   
    bl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=bl_flipper.body,
        prefix="bl_flipper",
    )
   
    br_leg.sites[ModuleFaces.FRONT].attach_body(
        body=br_flipper.body,
        prefix="br_flipper",
    )
    return core
