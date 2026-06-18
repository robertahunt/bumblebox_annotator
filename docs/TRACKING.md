# Instance Tracking Feature

## Overview

The bee annotator now includes ByteTrack-inspired instance tracking to maintain consistent IDs across video frames. This feature automatically matches YOLO detections to existing annotations, preserving instance identities throughout the video.

## How It Works

### Tracking Algorithm

The tracker uses a two-stage matching approach:

1. **Stage 1 - High Confidence Matching**: Match detections with confidence ≥ 0.6 using IoU threshold of 0.6
2. **Stage 2 - Low Confidence Matching**: Match remaining detections using lower IoU threshold of 0.3

### IoU Calculation

- **Mask IoU** (default): More accurate, computed from overlapping pixels in segmentation masks
- **Box IoU**: Faster, computed from bounding box overlap

### Track Lifecycle

- **Active Tracks**: Matched in current frame
- **Lost Tracks**: Not matched for 1-5 frames (occlusion tolerance)
- **Terminated**: Exceeds 5 consecutive frames without match

## Usage

### Enabling Tracking

Tracking is **enabled by default**. To toggle:

1. Menu: `Model` → `Enable Instance Tracking` (checkbox)
2. Or use the tracking toolbar button (if available)

### Running YOLO with Tracking

1. Load a YOLO checkpoint via the YOLO toolbar
2. Navigate to a frame
3. Click "Run on Current Frame"
4. **With tracking enabled**:
   - New detections will match to previous frame's annotations
   - Matching instances keep their IDs
   - New instances get new unique IDs
5. **With tracking disabled**:
   - All detections get sequential IDs starting from next available

### Propagation with Tracking

When you propagate masks to the next frame:
- Color coding is maintained per instance ID
- Same bee will have the same color and ID throughout the video

### Best Practices

1. **Initialize with First Frame**: Annotate the first frame manually or with YOLO to establish initial IDs
2. **Sequential Processing**: Process frames in order for best tracking results
3. **Verify Matches**: Check that IDs are correctly maintained, especially after occlusions
4. **Manual Override**: You can manually edit IDs if tracking makes mistakes

## Configuration

### Default Settings

```python
{
    'iou_threshold_high': 0.6,   # High-confidence match threshold
    'iou_threshold_low': 0.3,    # Low-confidence match threshold
    'max_frames_lost': 5,        # Frames before track deletion
    'min_detection_conf': 0.25,  # Minimum YOLO confidence
    'use_mask_iou': True,        # Use mask IoU vs box IoU
    'match_strategy': 'hungarian',  # Hungarian vs greedy matching
}
```

### Adjusting Parameters

To modify tracking parameters, edit the InstanceTracker initialization in `gui/main_window.py`:

```python
config = {
    'iou_threshold_high': 0.7,  # More strict matching
    'max_frames_lost': 10,      # More occlusion tolerance
}
self.tracker = InstanceTracker(config=config)
```

## Workflow Examples

### Example 1: Full Video Tracking

1. Load project and video
2. Run YOLO on frame 0 → IDs 1, 2, 3 assigned
3. Navigate to frame 1
4. Run YOLO on frame 1 → Detections matched to IDs 1, 2, 3 if similar positions
5. Navigate to frame 2
6. Run YOLO on frame 2 → Continue matching...

### Example 2: Correcting Tracking Errors

If tracker assigns wrong ID:
1. Delete the incorrectly matched instance
2. Manually annotate with SAM2 using correct ID
3. Continue to next frame - tracker will use corrected ID

### Example 3: Handling New Bees Entering Frame

1. Frame N: Bees have IDs 1, 2, 3
2. Frame N+1: New bee enters
3. Run YOLO → Existing bees matched to 1, 2, 3; new bee gets ID 4

## Technical Details

### Data Structures

**Detection Object**:
```python
Detection(
    bbox=[x1, y1, x2, y2],  # Bounding box
    mask=ndarray,            # Binary mask (H, W)
    confidence=0.85,         # Detection confidence
    source='yolo',           # Source: 'yolo', 'sam2', 'manual', 'propagated'
    class_id=0               # Class ID
)
```

**Track Object**:
```python
Track(
    track_id=1,                      # Unique ID
    bbox=[x1, y1, x2, y2],          # Current bbox
    mask=ndarray,                    # Current mask
    last_seen_frame=42,              # Last matched frame
    source_history=['yolo', 'yolo'], # Detection sources
    confidence_history=[0.85, 0.82], # Confidence scores
    frames_lost=0                    # Consecutive frames without match
)
```

### Per-Video Tracking State

Each video maintains separate tracking state:
- Independent track IDs
- Separate color assignments
- Independent next_mask_id counter

This ensures IDs don't conflict when switching between videos.

## Troubleshooting

### Issue: IDs not maintained across frames
- **Solution**: Ensure tracking is enabled in Model menu
- **Solution**: Check that you're processing frames sequentially
- **Solution**: Try increasing `iou_threshold_low` for more lenient matching

### Issue: Too many new IDs assigned
- **Solution**: Lower `iou_threshold_high` for stricter matching
- **Solution**: Increase `max_frames_lost` for better occlusion handling

### Issue: Wrong bees being matched
- **Solution**: Use mask IoU instead of box IoU (more accurate)
- **Solution**: Manually correct and continue - tracker will learn from corrections

### Issue: Tracking slow on large videos
- **Solution**: Disable tracking for initial rough annotation
- **Solution**: Enable tracking only for final pass
- **Solution**: Use box IoU instead of mask IoU (faster)

## API Reference

### InstanceTracker Class

#### Methods

- `reset()`: Clear all tracking state
- `set_next_track_id(next_id)`: Set next available ID
- `match_detections_to_tracks(detections, frame_idx)`: Main matching function
- `get_track_info(track_id)`: Query track information
- `get_active_track_ids()`: List all active tracks

#### Parameters

See Configuration section above for parameter descriptions.

## Future Enhancements

Potential improvements:
- Kalman filtering for motion prediction
- ReID features for appearance-based matching  
- Batch processing: Run YOLO+tracking on entire video
- Interactive track editing UI
- Export tracking results for analysis
