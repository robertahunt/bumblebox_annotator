# Instance Tracking Implementation - Summary

## Files Created

### 1. `core/instance_tracker.py` (New)
Complete ByteTrack-inspired tracking module with:
- `Detection` dataclass - represents a detection in current frame
- `Track` dataclass - represents a tracked instance across frames
- `InstanceTracker` class - main tracking logic
  - Two-stage matching (high/low confidence)
  - Hungarian algorithm for optimal assignment
  - IoU-based matching (box and mask)
  - Track lifecycle management (active/lost/terminated)

### 2. `docs/TRACKING.md` (New)
Comprehensive documentation covering:
- How tracking works
- Usage instructions
- Configuration options
- Workflow examples
- Troubleshooting guide
- API reference

## Files Modified

### 1. `gui/main_window.py`
**Imports Added:**
- `from core.instance_tracker import InstanceTracker, Detection, Track`

**Initialization (`__init__`):**
- Added `self.tracker = InstanceTracker()`
- Added `self.video_trackers = {}` - per-video tracking state
- Added `self.tracking_enabled = True` - toggle flag

**New Methods:**
- `_get_or_create_tracker(video_id)` - get/create tracker per video
- `_yolo_results_to_detections(yolo_result, model)` - convert YOLO to Detection objects
- `_annotations_to_tracks(annotations)` - convert annotations to Track objects
- `toggle_tracking()` - UI callback for enabling/disabling tracking

**Modified Methods:**
- `run_yolo_inference()` - now uses tracker to match detections
  - Converts YOLO results to Detection objects
  - Matches detections to existing tracks
  - Preserves IDs for matched instances
  - Assigns new IDs for unmatched detections
  - Updates canvas with tracked IDs and colors
  
- `propagate_to_next_frame()` - now updates tracker state
  - Registers propagated masks with tracker
  - Updates track positions and metadata
  - Maintains tracking consistency across propagation

- `create_menus()` - added tracking toggle
  - Added "Enable Instance Tracking" checkbox in Model menu
  - Connected to `toggle_tracking()` callback

## Key Features

### 1. Automatic ID Matching
When running YOLO inference:
- Detections are matched to previous frame's annotations
- Matching instances keep their IDs
- New instances get unique IDs
- Works across frame gaps (occlusion tolerance)

### 2. Per-Video Tracking State
- Each video maintains independent tracker
- No ID conflicts between videos
- Automatic state management when switching videos

### 3. Color Persistence
- Instance colors maintained across frames
- Same bee = same color throughout video
- Makes visual tracking easier

### 4. Flexible Configuration
Default parameters (can be customized):
```python
{
    'iou_threshold_high': 0.6,   # High-confidence match
    'iou_threshold_low': 0.3,    # Low-confidence match  
    'max_frames_lost': 5,        # Occlusion tolerance
    'use_mask_iou': True,        # Use mask vs box IoU
    'match_strategy': 'hungarian' # Matching algorithm
}
```

### 5. Source Tracking
Each track records:
- Source history ('yolo', 'sam2', 'manual', 'propagated')
- Confidence history
- Frames since last seen
- Last seen frame index

## Usage Workflow

### Basic Workflow:
1. Load video/project
2. **Enable Tracking** (Model menu - enabled by default)
3. Run YOLO on frame 0 → IDs 1, 2, 3 assigned
4. Navigate to frame 1
5. Run YOLO on frame 1 → Detections matched to IDs 1, 2, 3
6. Continue through video...

### With Propagation:
1. Annotate frame 0 with YOLO
2. Propagate to frame 1 (IDs preserved)
3. Run YOLO on frame 1 to verify/refine
4. Tracker matches YOLO detections to propagated masks
5. Continue...

### Disabling Tracking:
1. Uncheck Model → "Enable Instance Tracking"
2. YOLO will assign sequential IDs without matching

## Technical Implementation

### Matching Algorithm (ByteTrack-inspired):

**Stage 1: High-Confidence Matching**
```
- Filter detections with confidence ≥ 0.6
- Compute IoU matrix (N_detections × N_tracks)
- Apply Hungarian matching
- Keep matches with IoU ≥ 0.6
```

**Stage 2: Low-Confidence Matching**
```
- Take remaining detections and tracks
- Compute IoU matrix
- Apply Hungarian matching  
- Keep matches with IoU ≥ 0.3
```

**Post-Processing:**
```
- Update matched tracks
- Create new tracks for unmatched detections
- Increment lost counter for unmatched tracks
- Delete tracks lost > 5 frames
```

### IoU Computation:

**Mask IoU** (default, more accurate):
```python
intersection = (mask1 & mask2).sum()
union = (mask1 | mask2).sum()
iou = intersection / union
```

**Box IoU** (faster, less accurate):
```python
x1 = max(box1.x1, box2.x1)
y1 = max(box1.y1, box2.y1)
x2 = min(box1.x2, box2.x2)
y2 = min(box1.y2, box2.y2)

intersection = (x2-x1) * (y2-y1)
area1 = box1.area
area2 = box2.area
union = area1 + area2 - intersection
iou = intersection / union
```

## Benefits

1. **Consistency**: Same bee gets same ID throughout video
2. **Automation**: Reduces manual ID assignment
3. **Flexibility**: Works with YOLO, SAM2, manual annotations
4. **Robustness**: Handles occlusions, entries, exits
5. **Visual Clarity**: Color coding makes tracking obvious
6. **No Conflicts**: Per-video state prevents ID reuse

## Future Enhancements

Potential improvements:
- [ ] Motion prediction with Kalman filtering
- [ ] Appearance-based ReID features
- [ ] Batch video processing
- [ ] Interactive track editing UI
- [ ] Multi-class tracking
- [ ] Track quality metrics
- [ ] Export tracking results

## Testing

To test the implementation:

1. **Basic Test:**
   - Load a video with multiple bees
   - Run YOLO on frame 0
   - Note the assigned IDs
   - Run YOLO on frame 1
   - Verify IDs are preserved for same bees

2. **Occlusion Test:**
   - Find frames where bee is temporarily hidden
   - Run YOLO before occlusion
   - Skip 2-3 frames
   - Run YOLO after occlusion
   - Verify ID is recovered

3. **New Entry Test:**
   - Run YOLO on frame with N bees
   - Run YOLO on frame with N+1 bees (new bee enters)
   - Verify existing bees keep IDs, new bee gets new ID

4. **Toggle Test:**
   - Run YOLO with tracking enabled → note IDs
   - Disable tracking
   - Run YOLO again → note IDs (should be different)
   - Re-enable tracking
   - Run YOLO → should resume tracking

## Dependencies

No new external dependencies required:
- `scipy` - for `linear_sum_assignment` (Hungarian algorithm)
- `numpy` - already in project
- `cv2` - already in project

All dependencies should already be installed in the project environment.
