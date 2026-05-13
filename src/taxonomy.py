"""Skill and coordination taxonomy for bimanual robot manipulation.

The taxonomy is the *contract* between the LLM, the dataset, the retriever,
and the training-mix recommender. Adding new skills should be done here
and nowhere else, so prompts, schemas, and indices stay consistent.
"""
from __future__ import annotations

from typing import Dict, List, Set


SKILLS: List[str] = [
    "delicate_grasp",
    "bimanual_stabilize",
    "bimanual_handoff",
    "hold_and_manipulate",
    "contact_rich_motion",
    "force_control",
    "tool_use",
    "pouring",
    "sprinkling",
    "stirring",
    "wiping",
    "folding",
    "pulling",
    "twisting",
    "opening",
    "precision_placement",
    "container_interaction",
    "liquid_handling",
    "deformable_object_handling",
    "sequencing",
    "recovery_behavior",
]


COORDINATION_TYPES: List[str] = [
    "single_arm",
    "independent_dual_arm",
    "leader_follower",
    "stabilize_then_act",
    "simultaneous_bimanual",
    "handoff",
    "symmetric_motion",
    "asymmetric_motion",
    "support_and_manipulate",
]


SKILL_DESCRIPTIONS: Dict[str, str] = {
    "delicate_grasp": "Low-force, controlled grasp suitable for fragile or deformable objects.",
    "bimanual_stabilize": "One arm holds an object steady so another agent or arm can act on it.",
    "bimanual_handoff": "Object is transferred between the two arms.",
    "hold_and_manipulate": "One hand holds the object while the other performs the primary manipulation.",
    "contact_rich_motion": "Motion involving sustained or transient contact, such as tapping, pressing, or breaking.",
    "force_control": "Compliant control where applied force/torque is regulated rather than position alone.",
    "tool_use": "Use of an external tool (spoon, knife, sponge, shaker) as an end-effector extension.",
    "pouring": "Controlled flow of contents from one container to another.",
    "sprinkling": "Discrete, low-volume dispensing motion (salt, seasoning, particulates).",
    "stirring": "Cyclic motion of a tool inside a container to mix contents.",
    "wiping": "Pressed sweeping motion across a surface, typically with a cloth or sponge.",
    "folding": "Controlled deformation of a flexible object across one or more creases.",
    "pulling": "Linear extraction motion along an object's primary axis (drawer, lid).",
    "twisting": "Rotational motion about an object's axis (cap, knob, screw).",
    "opening": "Composite motion that exposes a container or compartment.",
    "precision_placement": "Sub-centimeter accurate placement, often into a target slot or container.",
    "container_interaction": "Reaching into, around, or above a container while respecting its geometry.",
    "liquid_handling": "Manipulation of liquids with awareness of spillage and slosh dynamics.",
    "deformable_object_handling": "Manipulation of cloth, dough, or other non-rigid objects.",
    "sequencing": "Multi-step ordering of sub-skills, where step k depends on step k-1.",
    "recovery_behavior": "Reactive correction after a slip, drop, or unexpected contact.",
}


COORDINATION_DESCRIPTIONS: Dict[str, str] = {
    "single_arm": "Only one arm is task-relevant; the other is idle or out of the way.",
    "independent_dual_arm": "Both arms work, but on independent sub-tasks with no tight coupling.",
    "leader_follower": "One arm leads (sets pose/timing); the other tracks or reacts.",
    "stabilize_then_act": "One arm stabilizes the workpiece, then the other arm executes the primary action.",
    "simultaneous_bimanual": "Both arms act on the same workpiece at the same time with shared timing.",
    "handoff": "Object physically transitions from one arm to the other.",
    "symmetric_motion": "Both arms execute mirrored or near-identical trajectories.",
    "asymmetric_motion": "Arms execute distinct trajectories with different roles.",
    "support_and_manipulate": "One arm provides passive support/contact; the other manipulates actively.",
}


SKILL_SET: Set[str] = set(SKILLS)
COORDINATION_SET: Set[str] = set(COORDINATION_TYPES)


SKILL_FAILURE_MODES: Dict[str, List[str]] = {
    "delicate_grasp": [
        "object slip due to insufficient grip force",
        "object crush from over-force during pickup",
    ],
    "bimanual_stabilize": [
        "support arm instability causes workpiece to drift",
        "support arm releases too early before primary action completes",
    ],
    "bimanual_handoff": [
        "object dropped at the midline transfer point",
        "double-grasp deadlock where neither arm releases",
    ],
    "hold_and_manipulate": [
        "holding arm rotates the object while the other arm acts on it",
        "holding force decays as manipulation forces increase",
    ],
    "contact_rich_motion": [
        "premature or excessive contact damages the workpiece",
        "loss of contact mid-motion truncates the skill",
    ],
    "force_control": [
        "torque overshoot during compliant phases",
        "stiction causes step-changes in applied force",
    ],
    "tool_use": [
        "tool slips out of the end-effector during application",
        "tool tip misaligned with target reference frame",
    ],
    "pouring": [
        "stream misaligned with the receptacle, spilling outside",
        "over-pour because tilt rate is not regulated against fill level",
    ],
    "sprinkling": [
        "no particles dispensed because tap impulse is too soft",
        "clumped dispense because tap rhythm is irregular",
    ],
    "stirring": [
        "stirring trajectory leaves the bowl perimeter",
        "spoon collides with bowl rim instead of tracking the contents",
    ],
    "wiping": [
        "cloth lifts off surface mid-stroke and skips coverage",
        "uneven pressure leaves untouched regions",
    ],
    "folding": [
        "creases land in the wrong location due to bad grip points",
        "fabric slips out of one gripper mid-fold and unfolds",
    ],
    "pulling": [
        "drawer/handle slips out of grip under load",
        "pull direction drifts off the rail axis and binds",
    ],
    "twisting": [
        "cap stripped because torque exceeds object tolerance",
        "rotation axis misaligned with the object axis",
    ],
    "opening": [
        "lid sticks because torque ramp is too aggressive",
        "lid not fully separated before retract phase begins",
    ],
    "precision_placement": [
        "placement misses target slot tolerance and rolls off",
        "object released at non-zero velocity, bouncing out of place",
    ],
    "container_interaction": [
        "end-effector collides with container rim on entry",
        "object dropped above container edge instead of inside",
    ],
    "liquid_handling": [
        "slosh exceeds container capacity during transit",
        "container tilted, causing leak before pour phase",
    ],
    "deformable_object_handling": [
        "object collapses into an unrecoverable configuration",
        "grip point drifts on the deformable surface mid-action",
    ],
    "sequencing": [
        "step k-1 finishes in a state that step k cannot consume",
        "arm desynchronization between sequenced phases",
    ],
    "recovery_behavior": [
        "no reactive correction after a slip; episode terminates failed",
        "over-correction drives the system into a worse state",
    ],
}


SKILL_GAP_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "delicate_grasp": {
        "why": "Fragile and deformable objects need a low-force, compliant pickup the policy has not learned.",
        "what_fails": "the grasp either crushes the object or fails to lift it",
        "minimal_fix": "20-30 demos of delicate pickups on fragile or thin items (eggs, wafers, glass)",
    },
    "bimanual_stabilize": {
        "why": "One arm must keep the workpiece pose-fixed while the other acts on it.",
        "what_fails": "the workpiece drifts and the primary action loses its reference frame",
        "minimal_fix": "15-25 demos where one arm holds an object steady throughout the episode",
    },
    "bimanual_handoff": {
        "why": "Object transfer between arms needs synchronized open/close timing.",
        "what_fails": "object dropped or double-grasped at the midline waypoint",
        "minimal_fix": "20 demos of midline handoffs with varied object shapes",
    },
    "hold_and_manipulate": {
        "why": "One arm must apply a passive holding wrench while the other does the active task.",
        "what_fails": "the held object slips or rotates as the other arm applies forces",
        "minimal_fix": "15-25 demos where one arm holds while the other manipulates",
    },
    "contact_rich_motion": {
        "why": "Tapping, breaking, and pressing motions need controlled transient contact.",
        "what_fails": "contact arrives too hard (damage) or too soft (no effect)",
        "minimal_fix": "20-30 demos with explicit tap/press/break events",
    },
    "force_control": {
        "why": "Compliance is required when position-only control would jam or damage.",
        "what_fails": "torque overshoot or stiction-driven step changes during contact phases",
        "minimal_fix": "compliance-labelled demos with varied stiffness profiles (15-25 episodes)",
    },
    "tool_use": {
        "why": "The end-effector must treat an external tool as a kinematic extension.",
        "what_fails": "tool slips, misaligns, or is held with the wrong grip pose",
        "minimal_fix": "20-30 demos using the same tool family as the target task",
    },
    "pouring": {
        "why": "Liquid transfer needs simultaneous tilt control and stream targeting.",
        "what_fails": "spill outside the receptacle or over-pour past the fill level",
        "minimal_fix": "20-30 pour demos across cup/bottle/pitcher types",
    },
    "sprinkling": {
        "why": "Discrete particle dispensing needs a tap impulse the policy has not seen.",
        "what_fails": "either nothing dispenses or particles dump in clumps",
        "minimal_fix": "20-30 demos of shaker / packet tap dispensing onto a target surface",
    },
    "stirring": {
        "why": "Cyclic in-container motion must respect bowl geometry and contents.",
        "what_fails": "spoon collides with rim or escapes the bowl perimeter",
        "minimal_fix": "15-25 stir demos in bowls of varied diameters",
    },
    "wiping": {
        "why": "Sustained surface contact with even pressure is required across a swept path.",
        "what_fails": "cloth lifts off and leaves untouched regions",
        "minimal_fix": "15-25 wipe demos with cloth/sponge across flat and curved surfaces",
    },
    "folding": {
        "why": "Cloth folds need correct grip-point selection and a sequenced fold path.",
        "what_fails": "creases land in the wrong place or the fabric unfolds itself",
        "minimal_fix": "20-30 fold demos on towels and shirts with varied initial poses",
    },
    "pulling": {
        "why": "Linear extraction along an object axis under load requires axis tracking.",
        "what_fails": "handle slips out of grip or pull binds off-axis",
        "minimal_fix": "10-20 demos of drawer / lid / handle pulls",
    },
    "twisting": {
        "why": "Rotational motion about an object axis with controlled torque.",
        "what_fails": "rotation axis drifts or torque exceeds object tolerance",
        "minimal_fix": "20-30 demos of cap / lid / knob rotations with varied torque thresholds",
    },
    "opening": {
        "why": "Composite skill that exposes a container or compartment.",
        "what_fails": "lid sticks or is not fully separated before retract",
        "minimal_fix": "15-25 opening demos across jar, box, drawer types",
    },
    "precision_placement": {
        "why": "Sub-centimeter placement into a target requires final-approach control.",
        "what_fails": "object released off-target or with non-zero velocity",
        "minimal_fix": "20-30 placement demos into varied receptacles",
    },
    "container_interaction": {
        "why": "Reaching into / above a container without colliding with its rim.",
        "what_fails": "rim collisions or releases that miss the container interior",
        "minimal_fix": "15-25 demos of in-container reaches with multiple container shapes",
    },
    "liquid_handling": {
        "why": "Transit and tilt must respect slosh and fill dynamics.",
        "what_fails": "spill in transit or leak from over-tilted source",
        "minimal_fix": "20 demos of carrying and pouring filled containers",
    },
    "deformable_object_handling": {
        "why": "Cloth, dough, and paper can collapse to unrecoverable states.",
        "what_fails": "object enters a configuration the policy cannot exit",
        "minimal_fix": "20-30 demos with cloth / dough / paper across grip points",
    },
    "sequencing": {
        "why": "Multi-step tasks require step k to leave a state step k+1 can consume.",
        "what_fails": "arms desynchronize or step k-1 ends in an unusable state",
        "minimal_fix": "15-25 multi-stage demos where bimanual sub-steps are stitched",
    },
    "recovery_behavior": {
        "why": "Real demonstrations slip; the policy needs reactive correction.",
        "what_fails": "any single slip terminates the episode in failure",
        "minimal_fix": "10-20 demos that include an intentional perturbation and recovery",
    },
}


COORDINATION_FAILURE_MODES: Dict[str, List[str]] = {
    "stabilize_then_act": ["support arm releases before action completes"],
    "simultaneous_bimanual": ["arm desynchronization across the shared timeline"],
    "leader_follower": ["follower arm lags or drifts off leader trajectory"],
    "handoff": ["object dropped at the midline transfer point"],
    "symmetric_motion": ["asymmetric drift breaks the mirror constraint"],
    "asymmetric_motion": ["task-irrelevant arm interferes with the active arm"],
    "support_and_manipulate": ["passive support contact lost mid-task"],
    "independent_dual_arm": ["arms collide while executing independent sub-tasks"],
    "single_arm": ["unused arm intrudes into the workspace"],
}


def failure_modes_from_skills(skills: List[str]) -> List[str]:
    """Aggregate canonical failure modes for a list of skills, dedup-preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for s in skills:
        for fm in SKILL_FAILURE_MODES.get(s, []):
            if fm not in seen:
                seen.add(fm)
                out.append(fm)
    return out


def failure_modes_from_coordination(coords: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for c in coords:
        for fm in COORDINATION_FAILURE_MODES.get(c, []):
            if fm not in seen:
                seen.add(fm)
                out.append(fm)
    return out


KEYWORD_TO_SKILL: Dict[str, List[str]] = {
    "egg": ["delicate_grasp", "contact_rich_motion", "bimanual_stabilize", "sequencing"],
    "crack": ["contact_rich_motion", "force_control", "bimanual_stabilize", "sequencing"],
    "salt": ["sprinkling", "tool_use", "container_interaction"],
    "season": ["sprinkling", "tool_use"],
    "sprinkle": ["sprinkling"],
    "pepper": ["sprinkling", "tool_use"],
    "pour": ["pouring", "liquid_handling", "container_interaction"],
    "juice": ["pouring", "liquid_handling", "hold_and_manipulate"],
    "water": ["pouring", "liquid_handling", "hold_and_manipulate"],
    "cup": ["hold_and_manipulate", "container_interaction"],
    "bottle": ["hold_and_manipulate", "twisting", "container_interaction"],
    "jar": ["bimanual_stabilize", "twisting", "force_control", "opening"],
    "lid": ["twisting", "opening", "force_control"],
    "cap": ["twisting", "opening", "force_control"],
    "open": ["opening", "force_control"],
    "close": ["opening", "force_control"],
    "fold": ["folding", "deformable_object_handling", "bimanual_stabilize"],
    "towel": ["folding", "deformable_object_handling"],
    "cloth": ["folding", "deformable_object_handling"],
    "shirt": ["folding", "deformable_object_handling"],
    "wipe": ["wiping", "tool_use", "contact_rich_motion"],
    "clean": ["wiping", "tool_use", "contact_rich_motion"],
    "sponge": ["tool_use", "wiping"],
    "stir": ["stirring", "tool_use", "container_interaction"],
    "mix": ["stirring", "tool_use", "container_interaction"],
    "bowl": ["container_interaction", "hold_and_manipulate"],
    "pan": ["container_interaction", "tool_use"],
    "tray": ["bimanual_stabilize", "symmetric_motion", "hold_and_manipulate"],
    "drawer": ["pulling", "force_control"],
    "pull": ["pulling", "force_control"],
    "twist": ["twisting", "force_control"],
    "tighten": ["twisting", "force_control"],
    "screw": ["twisting", "force_control", "precision_placement"],
    "assemble": ["precision_placement", "bimanual_handoff", "force_control", "sequencing"],
    "insert": ["precision_placement", "force_control"],
    "place": ["precision_placement"],
    "stack": ["precision_placement", "sequencing"],
    "block": ["precision_placement", "delicate_grasp"],
    "fragile": ["delicate_grasp", "force_control"],
    "egg_carton": ["delicate_grasp"],
    "peel": ["contact_rich_motion", "force_control", "deformable_object_handling"],
    "sticker": ["contact_rich_motion", "deformable_object_handling"],
    "cut": ["tool_use", "bimanual_stabilize", "force_control"],
    "knife": ["tool_use", "force_control"],
    "spoon": ["tool_use", "stirring"],
    "shaker": ["tool_use", "sprinkling"],
    "pack": ["precision_placement", "container_interaction", "sequencing"],
    "container": ["container_interaction", "precision_placement"],
    "transfer": ["bimanual_handoff", "hold_and_manipulate"],
    "handoff": ["bimanual_handoff"],
    "pass": ["bimanual_handoff"],
    "hold": ["hold_and_manipulate", "bimanual_stabilize"],
    "stabilize": ["bimanual_stabilize", "support_and_manipulate"],
    "support": ["support_and_manipulate"],
    "tap": ["contact_rich_motion", "force_control"],
}


KEYWORD_TO_COORDINATION: Dict[str, List[str]] = {
    "stabilize": ["stabilize_then_act", "support_and_manipulate"],
    "hold": ["stabilize_then_act", "support_and_manipulate"],
    "while": ["asymmetric_motion", "leader_follower"],
    "and": ["asymmetric_motion"],
    "two hand": ["symmetric_motion", "simultaneous_bimanual"],
    "both hand": ["symmetric_motion", "simultaneous_bimanual"],
    "tray": ["symmetric_motion", "simultaneous_bimanual"],
    "handoff": ["handoff"],
    "pass": ["handoff"],
    "transfer": ["handoff"],
    "fold": ["simultaneous_bimanual", "symmetric_motion"],
    "assemble": ["simultaneous_bimanual", "asymmetric_motion"],
    "pour": ["asymmetric_motion", "leader_follower"],
    "sprinkle": ["asymmetric_motion", "leader_follower"],
    "wipe": ["stabilize_then_act", "asymmetric_motion"],
    "stir": ["stabilize_then_act"],
    "open": ["stabilize_then_act"],
    "twist": ["stabilize_then_act"],
    "crack": ["stabilize_then_act", "asymmetric_motion"],
}


def is_known_skill(skill: str) -> bool:
    return skill in SKILL_SET


def is_known_coordination(coord: str) -> bool:
    return coord in COORDINATION_SET


def filter_to_known_skills(skills: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in skills:
        s_norm = s.strip().lower().replace(" ", "_").replace("-", "_")
        if s_norm in SKILL_SET and s_norm not in seen:
            seen.add(s_norm)
            out.append(s_norm)
    return out


def filter_to_known_coordination(coords: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for c in coords:
        c_norm = c.strip().lower().replace(" ", "_").replace("-", "_")
        if c_norm in COORDINATION_SET and c_norm not in seen:
            seen.add(c_norm)
            out.append(c_norm)
    return out
