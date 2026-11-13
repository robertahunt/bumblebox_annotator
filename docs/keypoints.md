# Keypoints (optional)

Keypoints are landmarks on each bee, such as head, thorax, abdomen tip, antenna tips and leg tips.
They are saved with the bee's mask and exported to COCO. They are **off by default**. A project
that never turns them on looks and behaves exactly as before.

This is phase 1 of the SLEAP plan: placing keypoints by hand. It needs no extra packages.
Exporting to and importing from SLEAP (`.slp`), and training or predicting with SLEAP from
the app, come later.

## Turning keypoints on

**Project > Keypoints...** opens the settings:

- **Use keypoints in this project:** the on/off switch.
- **Keypoints (in order):** one name per line, in clicking order. The default has 11 points:
  the body axis first (`head`, `thorax`, `abdomen_tip`), then once around the bee:
  `left_antenna_tip`, `left_front_leg_tip`, `left_mid_leg_tip`, `left_hind_leg_tip`,
  `right_hind_leg_tip`, `right_mid_leg_tip`, `right_front_leg_tip`, `right_antenna_tip`.
  Antennae and legs are often hidden; skip those (right-click). A leg tip is the end of the
  leg, where it meets the surface. Use letters, digits and underscores.
  **Left and right are the bee's own left and right**, not the image's. Names that start
  with `left_` and `right_` are recorded as mirror pairs (`symmetries`), which pose tools
  such as SLEAP use to swap the two sides when they flip images during training.
  **Restore Bee Defaults** puts the default list back. A project keeps the list it was
  saved with, so a project set up with the earlier five-point default (no legs) gets the
  leg tips only when you press this and save.
- **Skeleton lines:** pairs to connect on screen and in the COCO skeleton, for example
  `head-thorax, thorax-abdomen_tip, head-left_antenna_tip, thorax-left_front_leg_tip`. The
  default joins the head to each antenna tip and the thorax to each leg tip.
- **Fill from ArUco tag:** the keypoint that **Thorax from ArUco Tags** sets to each tag's
  centre. The default is `thorax`.

The settings are stored under `"keypoints"` in `annotations/project.json`. Like the
annotation policy, they can only be changed in the coordinator's full project, not in a
published backup or a contributor's assigned workspace.

> **Change settings between assignment rounds.** Once a changed `project.json` is published,
> contributors' outstanding assigned work can only be submitted for coordinator review.
> This applies to any project-settings change.

Turning keypoints off, or removing a name, never deletes saved keypoints. They stay in the
frame files and simply aren't shown or exported.

## Placing keypoints

When keypoints are on, a **Keypoints** tool joins the editing tools. While it is active, the
keypoint options (Next, Place as hidden, Show keypoints, Thorax from ArUco Tags, ...) appear
under the tool row.

1. Choose the **Keypoints** tool.
2. Click a bee. The click selects it and places its first missing keypoint.
3. Keep clicking. Each click places the next keypoint, shown in **Next:**, for that same bee,
   even when the click lands on a neighbouring bee (for example an antenna tip). When the bee
   is complete, a click on another bee starts that one.
4. When the next point isn't visible, **right-click anywhere** (or press N) to skip it. It is
   left unlabelled, and it isn't offered again for this bee. Right-clicking a bee you haven't
   started starts it and skips its first point. Leave a point unlabelled rather
   than guessing. Use **hidden** only for a point whose position you can still tell, for
   example a tip just under a neighbour's wing.
5. The bee is done once every point is placed or skipped, so the next click starts another
   bee. Esc finishes a bee early.
6. Change **Next:** at any time to redo a particular point, including a skipped one. Coming
   back to a bee later offers its skipped points again.

| Action | How |
|---|---|
| Move a point | Drag it (any visible bee) |
| Place one point as hidden (occluded) | Shift+click where it goes |
| Switch an existing point between hidden and visible | Shift+click it, or right-click it > **Mark Hidden/Visible** |
| Place every following point as hidden | Tick **Place as hidden** (H); untick to stop |
| Skip the next point (leave it unlabelled) | Right-click anywhere that isn't a point, or N |
| Point menu: hidden/visible, delete, finish or clear the bee | Right-click a point, or Shift+right-click anywhere |
| Delete a point | Hover and press Backspace/Delete, or right-click it > **Delete** |
| Undo the last point placed, or the last skip | Backspace |
| Finish the bee early | Esc, or right-click a point > **Finish Bee** |
| Undo/redo keypoint edits on this frame | Ctrl+Z / Ctrl+Shift+Z while the Keypoints tool is active |
| Thorax from every tag on the frame | **Thorax from ArUco Tags** (only fills missing points; one undo step) |

Points are drawn at a fixed screen size. Every `left_` point is green and every `right_`
point pink, so a left/right swap stands out. Hidden points are drawn as dashed rings, and
the bee being labelled is outlined in white. Keypoints follow the bee's visibility: they
hide with the bee class switch, with an individual bee switch, and with Space.

## Stored format

Each bee in the frame JSON may carry:

```json
"keypoints": {"head": [412.5, 233.0, 2], "thorax": [430.0, 251.5, 2], "abdomen_tip": [455.0, 280.0, 1]}
```

Coordinates are full-resolution image pixels. The third value follows COCO: 2 = visible,
1 = labelled but hidden. Points that aren't labelled are omitted.

- **Merging IDs** (ArUco reassignment) keeps the target bee's points and fills in any names
  it lacks from the merged bee.
- **Splitting** a bee leaves its keypoints on the original instance.
- **Edit tracking:** keypoint edits count as edits. They update `last_modified_by`, and they
  turn a bee frame marked finished into "changed". Frames without keypoints keep their
  existing signatures.

## COCO export

With keypoints on, the bee category gets COCO `keypoints` (names) and a 1-based `skeleton`.
Every bee annotation gets `keypoints` (flat `x, y, v` triples in the configured order, with
`0, 0, 0` when unlabelled) and `num_keypoints`. The settings are recorded in
`info.keypoints`, including the left/right `symmetries`. With keypoints off, exports are
unchanged.

## Known limits (phase 1)

- Re-running model inference on a frame replaces its bees, which drops their keypoints,
  markers and other per-bee fields.
- The submission comparison preview draws masks and boxes only. A conflict that differs
  only in keypoints shows two identical panes, although it is still detected.
- YOLO raster/pose exports don't include keypoints yet.
