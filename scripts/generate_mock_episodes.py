"""Reproducible generator for the mock bimanual episode dataset.

Run from repo root:

    python scripts/generate_mock_episodes.py

Produces ``data/mock_episodes.json`` with ~180 episodes spanning the full
skill / coordination taxonomy. Each template defines a task family with
canonical skills and natural language variations, then we sample variants
to keep the dataset realistic but deterministic.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.taxonomy import COORDINATION_SET, SKILL_SET  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "mock_episodes.json"

random.seed(20260510)


TEMPLATES: List[Dict] = [
    {
        "family": "open_jar",
        "task_name": "open a jar",
        "descriptions": [
            "left arm stabilizes the jar on the table while the right arm twists the lid counterclockwise",
            "right arm presses jar to the surface; left arm twists the cap until it loosens",
            "one arm braces the glass jar; the other arm applies torque to the screw lid",
        ],
        "objects": [["jar", "lid"], ["glass jar", "metal lid"], ["pickle jar", "lid"]],
        "skills": ["bimanual_stabilize", "hold_and_manipulate", "twisting", "force_control", "opening"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.78, 0.94),
        "count": 9,
    },
    {
        "family": "tighten_cap",
        "task_name": "tighten the cap on a bottle",
        "descriptions": [
            "left hand holds bottle upright while right hand twists cap clockwise to tighten",
            "right arm grasps bottle base; left arm rotates cap to seal it",
        ],
        "objects": [["bottle", "cap"], ["water bottle", "cap"]],
        "skills": ["hold_and_manipulate", "twisting", "force_control"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.80, 0.95),
        "count": 6,
    },
    {
        "family": "pull_drawer",
        "task_name": "open a drawer",
        "descriptions": [
            "right arm grasps the drawer handle and pulls outward; left arm steadies the cabinet front",
            "one arm pulls the drawer open along its rail; the other arm rests against the cabinet",
        ],
        "objects": [["drawer", "handle"], ["cabinet", "drawer", "handle"]],
        "skills": ["pulling", "force_control", "opening"],
        "coordination": "support_and_manipulate",
        "quality_range": (0.75, 0.92),
        "count": 6,
    },
    {
        "family": "fold_towel",
        "task_name": "fold a towel in half",
        "descriptions": [
            "both arms grasp the two corners of the towel and bring them together symmetrically",
            "left arm pins one edge of the towel; right arm folds the opposite edge over",
            "two arms execute a symmetric fold along the long axis of the cloth",
        ],
        "objects": [["towel"], ["hand towel"], ["cloth"]],
        "skills": ["folding", "deformable_object_handling", "bimanual_stabilize", "sequencing"],
        "coordination": "symmetric_motion",
        "quality_range": (0.70, 0.90),
        "count": 8,
    },
    {
        "family": "fold_shirt",
        "task_name": "fold a shirt",
        "descriptions": [
            "both arms execute a symmetric fold along the shirt's vertical axis; arms move in mirrored trajectories",
            "left arm tucks the sleeve in; right arm mirrors the motion on the opposite side",
        ],
        "objects": [["shirt"], ["t-shirt"]],
        "skills": ["folding", "deformable_object_handling", "bimanual_stabilize", "sequencing"],
        "coordination": "simultaneous_bimanual",
        "quality_range": (0.65, 0.88),
        "count": 6,
    },
    {
        "family": "pour_juice",
        "task_name": "pour juice into a cup",
        "descriptions": [
            "left arm holds the cup steady on the table while right arm tilts the carton to pour",
            "one arm holds the receptacle; the other tilts the carton until liquid flows",
            "right arm pours juice from the bottle; left arm stabilizes the cup to prevent tipping",
        ],
        "objects": [["cup", "carton"], ["glass", "bottle", "juice"], ["cup", "juice carton"]],
        "skills": ["pouring", "liquid_handling", "hold_and_manipulate", "container_interaction"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.72, 0.93),
        "count": 9,
    },
    {
        "family": "pour_water",
        "task_name": "pour water into a glass while holding the glass",
        "descriptions": [
            "left arm grips the glass on the counter; right arm tilts the pitcher and pours water in",
            "one arm holds the glass to keep it from sliding; the other pours from the kettle",
        ],
        "objects": [["glass", "pitcher"], ["glass", "kettle"]],
        "skills": ["pouring", "liquid_handling", "hold_and_manipulate"],
        "coordination": "leader_follower",
        "quality_range": (0.74, 0.92),
        "count": 7,
    },
    {
        "family": "stir_bowl",
        "task_name": "stir contents in a bowl",
        "descriptions": [
            "left arm holds the bowl on the counter while right arm stirs with a spoon",
            "right arm rotates the spoon inside the bowl; left arm stabilizes the bowl rim",
        ],
        "objects": [["bowl", "spoon"], ["mixing bowl", "wooden spoon"]],
        "skills": ["stirring", "tool_use", "bimanual_stabilize", "container_interaction"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.76, 0.93),
        "count": 7,
    },
    {
        "family": "wipe_table",
        "task_name": "wipe a table while holding an object steady",
        "descriptions": [
            "left arm presses a small object to the table to keep it in place; right arm wipes around it with a cloth",
            "one arm holds the cup steady; the other arm wipes the table surface with a sponge",
        ],
        "objects": [["table", "cloth", "cup"], ["table", "sponge", "object"]],
        "skills": ["wiping", "tool_use", "contact_rich_motion", "bimanual_stabilize"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.72, 0.91),
        "count": 7,
    },
    {
        "family": "clean_pan",
        "task_name": "clean a pan with a sponge",
        "descriptions": [
            "left arm holds the pan handle to brace it; right arm scrubs the pan with a sponge using contact-rich strokes",
            "one arm stabilizes the pan; the other applies the sponge in circular wiping motions",
        ],
        "objects": [["pan", "sponge"], ["frying pan", "sponge"]],
        "skills": ["wiping", "tool_use", "contact_rich_motion", "force_control", "bimanual_stabilize"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.70, 0.90),
        "count": 6,
    },
    {
        "family": "transfer_handoff",
        "task_name": "hand off an object between hands",
        "descriptions": [
            "left arm grasps an apple and passes it to the right arm at a midline waypoint",
            "right arm receives the cube from the left arm; both arms close grip simultaneously during transfer",
        ],
        "objects": [["apple"], ["cube"], ["small box"]],
        "skills": ["bimanual_handoff", "hold_and_manipulate", "precision_placement"],
        "coordination": "handoff",
        "quality_range": (0.78, 0.94),
        "count": 7,
    },
    {
        "family": "place_in_container",
        "task_name": "place an object into a container",
        "descriptions": [
            "right arm picks up an object and places it accurately into a small bin while left arm holds the bin",
            "left arm stabilizes the basket; right arm performs precision placement of the toy inside",
        ],
        "objects": [["toy", "bin"], ["block", "basket"], ["object", "container"]],
        "skills": ["precision_placement", "container_interaction", "hold_and_manipulate"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.78, 0.94),
        "count": 7,
    },
    {
        "family": "pick_fragile",
        "task_name": "pick up a fragile object",
        "descriptions": [
            "right arm executes a delicate grasp on an egg from the carton without crushing it",
            "left arm carefully lifts a glass figurine while applying minimal force",
            "delicate two-finger pinch on a thin wafer using compliant force control",
        ],
        "objects": [["egg", "carton"], ["glass figurine"], ["wafer"]],
        "skills": ["delicate_grasp", "force_control"],
        "coordination": "single_arm",
        "quality_range": (0.74, 0.92),
        "count": 8,
    },
    {
        "family": "stack_blocks",
        "task_name": "stack blocks on top of each other",
        "descriptions": [
            "right arm picks blocks one at a time and places them precisely on top of the previous block; left arm holds the base block steady",
            "left arm stabilizes the lower block; right arm performs precision placement of new blocks in sequence",
        ],
        "objects": [["wooden blocks"], ["cubes"]],
        "skills": ["precision_placement", "delicate_grasp", "sequencing", "bimanual_stabilize"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.74, 0.92),
        "count": 7,
    },
    {
        "family": "carry_tray",
        "task_name": "move a tray with two hands",
        "descriptions": [
            "both arms grasp the tray symmetrically and lift it together while keeping it level",
            "left and right arms execute a coordinated symmetric carry of the tray across the table",
        ],
        "objects": [["tray", "objects"], ["serving tray"]],
        "skills": ["bimanual_stabilize", "hold_and_manipulate"],
        "coordination": "symmetric_motion",
        "quality_range": (0.80, 0.95),
        "count": 6,
    },
    {
        "family": "peel_sticker",
        "task_name": "peel a sticker off a surface",
        "descriptions": [
            "left arm holds the box steady while right arm pinches and peels the sticker off using contact-rich motion",
            "one arm stabilizes the surface; the other arm peels the label using compliant force control",
        ],
        "objects": [["box", "sticker"], ["bottle", "label"]],
        "skills": ["contact_rich_motion", "force_control", "deformable_object_handling", "delicate_grasp", "bimanual_stabilize"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.68, 0.88),
        "count": 7,
    },
    {
        "family": "cut_object",
        "task_name": "cut an object with a knife while holding it steady",
        "descriptions": [
            "left arm holds the carrot down on the cutting board; right arm uses the knife to slice with controlled force",
            "one arm stabilizes the bread; the other arm cuts using a knife with rhythmic contact",
        ],
        "objects": [["carrot", "knife", "cutting board"], ["bread", "knife"]],
        "skills": ["tool_use", "bimanual_stabilize", "force_control", "contact_rich_motion"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.74, 0.93),
        "count": 7,
    },
    {
        "family": "pack_items",
        "task_name": "pack items into a container",
        "descriptions": [
            "right arm picks each object and inserts it precisely into the box; left arm holds the lid open",
            "left arm stabilizes the open container; right arm packs items in sequence into open slots",
        ],
        "objects": [["box", "objects"], ["container", "items"]],
        "skills": ["precision_placement", "container_interaction", "sequencing", "hold_and_manipulate"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.72, 0.92),
        "count": 7,
    },
    {
        "family": "sprinkle_seasoning",
        "task_name": "sprinkle seasoning over a dish",
        "descriptions": [
            "right arm shakes the salt shaker over the bowl while left arm holds the bowl steady",
            "one arm holds the dish; the other arm taps the seasoning shaker to dispense small amounts of salt",
        ],
        "objects": [["salt shaker", "bowl"], ["pepper shaker", "dish"], ["spice container", "pan"]],
        "skills": ["sprinkling", "tool_use", "container_interaction", "hold_and_manipulate"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.70, 0.90),
        "count": 7,
    },
    {
        "family": "tap_into_container",
        "task_name": "tap an object to release contents into a container",
        "descriptions": [
            "right arm taps the side of a small object over a bowl; left arm holds the bowl steady to catch contents",
            "one arm holds the bowl; the other taps a packet to dispense particles",
        ],
        "objects": [["packet", "bowl"], ["seed packet", "dish"]],
        "skills": ["contact_rich_motion", "force_control", "container_interaction", "hold_and_manipulate", "sprinkling"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.66, 0.88),
        "count": 5,
    },
    {
        "family": "assemble_parts",
        "task_name": "assemble two parts together",
        "descriptions": [
            "left arm holds part A steady while right arm aligns and inserts part B into it with compliant force",
            "both arms manipulate parts simultaneously to mate them along a tight tolerance",
            "one arm presents the receptacle; the other arm performs precision insertion",
        ],
        "objects": [["part_a", "part_b"], ["bracket", "screw"], ["block", "peg"]],
        "skills": ["precision_placement", "force_control", "bimanual_handoff", "sequencing", "contact_rich_motion"],
        "coordination": "simultaneous_bimanual",
        "quality_range": (0.70, 0.91),
        "count": 8,
    },
    {
        "family": "screw_in",
        "task_name": "screw a screw into a hole",
        "descriptions": [
            "left arm holds the bracket steady while right arm rotates the screwdriver",
            "one arm stabilizes the workpiece; the other arm executes torque-controlled twisting",
        ],
        "objects": [["screw", "screwdriver", "bracket"]],
        "skills": ["twisting", "force_control", "tool_use", "bimanual_stabilize", "precision_placement"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.72, 0.91),
        "count": 6,
    },
    {
        "family": "scoop_with_tool",
        "task_name": "scoop contents with a spoon",
        "descriptions": [
            "left arm holds the container steady while right arm scoops with a spoon",
            "one arm stabilizes the bowl; the other arm uses the spoon to scoop and lift contents",
        ],
        "objects": [["bowl", "spoon"], ["container", "scoop"]],
        "skills": ["tool_use", "container_interaction", "hold_and_manipulate", "bimanual_stabilize"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.74, 0.92),
        "count": 6,
    },
    {
        "family": "recover_drop",
        "task_name": "recover after dropping an object mid-task",
        "descriptions": [
            "after slipping the cup, the right arm reactively re-grasps it before it tips while the left arm continues stabilizing the tray",
            "object slips from grip; opposite arm catches and re-positions it before sequencing resumes",
        ],
        "objects": [["cup", "tray"], ["object"]],
        "skills": ["recovery_behavior", "delicate_grasp", "force_control", "sequencing"],
        "coordination": "leader_follower",
        "quality_range": (0.60, 0.85),
        "count": 5,
    },
    {
        "family": "open_box_lid",
        "task_name": "open a hinged box",
        "descriptions": [
            "left arm braces the box base; right arm lifts the hinged lid open",
            "one arm stabilizes the container; the other arm opens the lid along the hinge",
        ],
        "objects": [["box", "lid"]],
        "skills": ["opening", "pulling", "bimanual_stabilize", "hold_and_manipulate"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.78, 0.93),
        "count": 5,
    },
    {
        "family": "hold_while_drawing",
        "task_name": "draw on paper while holding the paper steady",
        "descriptions": [
            "left arm holds the paper flat on the table; right arm uses a pen to draw without slipping",
            "one arm stabilizes the sheet; the other arm performs precision tool use with a pen",
        ],
        "objects": [["paper", "pen"]],
        "skills": ["tool_use", "bimanual_stabilize", "precision_placement", "force_control"],
        "coordination": "asymmetric_motion",
        "quality_range": (0.72, 0.92),
        "count": 5,
    },
    {
        "family": "dual_carry_long",
        "task_name": "carry a long object with both hands",
        "descriptions": [
            "both arms grasp opposite ends of a rod and lift it while keeping it horizontal",
            "left and right arms hold each end of a stick and translate together across the workspace",
        ],
        "objects": [["rod"], ["stick"], ["broom handle"]],
        "skills": ["bimanual_stabilize", "hold_and_manipulate"],
        "coordination": "symmetric_motion",
        "quality_range": (0.78, 0.94),
        "count": 5,
    },
    {
        "family": "open_zipper",
        "task_name": "open a zipper on a pouch",
        "descriptions": [
            "left arm holds the pouch fabric taut; right arm pulls the zipper tab along the rail",
            "one arm tensions the bag; the other arm pulls the zipper to open it",
        ],
        "objects": [["pouch", "zipper"], ["bag", "zipper"]],
        "skills": ["pulling", "deformable_object_handling", "bimanual_stabilize", "force_control", "opening"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.66, 0.88),
        "count": 5,
    },
    {
        "family": "knead_dough",
        "task_name": "knead a piece of dough",
        "descriptions": [
            "both arms press and fold the dough on the table simultaneously with controlled force",
            "left and right arms perform symmetric kneading motions on a deformable mass",
        ],
        "objects": [["dough"]],
        "skills": ["deformable_object_handling", "force_control", "contact_rich_motion", "folding"],
        "coordination": "simultaneous_bimanual",
        "quality_range": (0.62, 0.86),
        "count": 4,
    },
    {
        "family": "press_button_while_hold",
        "task_name": "press a button while holding the device",
        "descriptions": [
            "left arm holds the device upright; right arm presses the button with controlled finger force",
            "one arm stabilizes the controller; the other arm taps the button cleanly",
        ],
        "objects": [["device", "button"], ["controller", "button"]],
        "skills": ["contact_rich_motion", "force_control", "hold_and_manipulate", "bimanual_stabilize"],
        "coordination": "stabilize_then_act",
        "quality_range": (0.78, 0.94),
        "count": 5,
    },
    {
        "family": "pour_then_stir",
        "task_name": "pour ingredients then stir them",
        "descriptions": [
            "right arm pours liquid from a measuring cup into a bowl, then switches to a spoon to stir while left arm holds the bowl",
            "sequenced bimanual task: pour first, then stir; left arm stabilizes the bowl across both phases",
        ],
        "objects": [["measuring cup", "bowl", "spoon"]],
        "skills": ["pouring", "stirring", "tool_use", "sequencing", "bimanual_stabilize", "container_interaction"],
        "coordination": "leader_follower",
        "quality_range": (0.66, 0.88),
        "count": 4,
    },
    {
        "family": "independent_dual",
        "task_name": "tidy two objects on a table",
        "descriptions": [
            "left arm picks up a cup and moves it to the rack; right arm independently picks a plate and places it on the shelf",
            "both arms execute independent pick-and-place actions on separate objects simultaneously",
        ],
        "objects": [["cup", "plate", "rack"], ["mug", "bowl"]],
        "skills": ["precision_placement", "hold_and_manipulate"],
        "coordination": "independent_dual_arm",
        "quality_range": (0.74, 0.92),
        "count": 4,
    },
]


def _validate_template(t: Dict) -> None:
    bad_skills = [s for s in t["skills"] if s not in SKILL_SET]
    if bad_skills:
        raise ValueError(f"Template {t['family']} has unknown skills: {bad_skills}")
    if t["coordination"] not in COORDINATION_SET:
        raise ValueError(f"Template {t['family']} has unknown coordination: {t['coordination']}")


def main() -> None:
    episodes: List[Dict] = []
    counter = 1

    for t in TEMPLATES:
        _validate_template(t)
        for _ in range(t["count"]):
            descr = random.choice(t["descriptions"])
            objs = random.choice(t["objects"])
            qmin, qmax = t["quality_range"]
            quality = round(random.uniform(qmin, qmax), 3)
            ep = {
                "episode_id": f"ep_{counter:04d}",
                "task_name": t["task_name"],
                "description": descr,
                "objects": list(objs),
                "skills": list(t["skills"]),
                "coordination_type": t["coordination"],
                "quality_score": quality,
            }
            episodes.append(ep)
            counter += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)

    print(f"Wrote {len(episodes)} episodes to {OUTPUT_PATH}")
    families = {}
    for ep in episodes:
        families[ep["task_name"]] = families.get(ep["task_name"], 0) + 1
    print("Distribution by task family:")
    for name, c in sorted(families.items(), key=lambda kv: -kv[1]):
        print(f"  {c:3d}  {name}")


if __name__ == "__main__":
    main()
