# Tracking Validation System

## Overview

The tracking validation system allows you to test and compare different tracking algorithms on annotated frame sequences. This helps you optimize tracking parameters and choose the best algorithm for your bee tracking application.

## Key Concepts

### Tracking Sequences

A **tracking sequence** is a set of consecutive frames (e.g., frame 100 → frame 101) where you have ground truth annotations. The system uses these to evaluate how well different tracking algorithms maintain consistent IDs across frames.

### How It Works

1. **Annotate consecutive frames** - Use your existing workflow to annotate 2+ consecutive frames
2. **Create tracking sequences** - Mark which frame pairs/groups to use for validation
3. **Run validation** - Test different tracking algorithms on these sequences
4. **Compare results** - View metrics showing which algorithm performs best

## Workflow

### Step 1: Annotate Ground Truth Frames

Use your existing annotation workflow:

1. Open a video in your project
2. Navigate to a frame (e.g., frame 100)
3. Annotate all bees (using YOLO, SAM2, or manual annotation)
4. Save the frame
5. Navigate to the next frame (frame 101)
6. Annotate all bees in this frame too
7. **Important**: Make sure instance IDs are consistent across frames (same bee = same ID)

**Tip**: Start with just 2 frames per sequence to keep annotation workload manageable.

### Step 2: Create Tracking Sequences

Once you have consecutive annotated frames:

1. Look for the **"Tracking Sequences"** panel on the right side
2. Navigate to one of your annotated frames
3. Click **"➕ New Sequence from Current Frame"**
4. In the dialog:
   - **Start Frame**: Set to your first annotated frame (e.g., 100)
   - **End Frame**: Set to your last consecutive annotated frame (e.g., 101)
   - **Notes**: Optionally add notes like "high activity" or "bees entering"
5. Click **"Create Sequence"**

The system will check if all frames have annotations and warn you if any are missing.

**Repeat this** for different videos or different time periods in the same video to create a diverse validation set.

### Step 3: Run Tracking Validation

1. In the **Tracking Sequences** panel, check the boxes next to sequences you want to validate
2. Click **"📊 Validate Selected Sequences"**  
   OR  
   Go to **Model → Validate Tracking Algorithms...**

3. In the validation dialog:
   - **Select a detection model** (YOLO checkpoint)
   - **Choose tracking algorithms** to test:
     - **ByteTrack**: Advanced two-stage matching (default)
     - **Simple IoU**: Baseline greedy IoU matching
     - **Centroid Distance**: Distance-based tracking
   - **Adjust parameters** for each algorithm if desired
   - **Select metrics** to calculate (MOTA, IDF1, etc.)

4. Click **"▶ Run Validation"**

### Step 4: View Results

During validation, you'll see:

- **Real-time metrics** for each algorithm
- **Live comparison plots** showing performance
- **Detailed logs** of the validation process

After completion:

- **Summary metrics** comparing all algorithms
- **Per-sequence breakdown** showing which sequences worked well/poorly
- **Saved results** in `<project>/tracking_validation/<timestamp>/`
  - CSV files with detailed metrics per algorithm
  - JSON summary with averages
  - Comparison plots

## Understanding Metrics

### MOTA (Multiple Object Tracking Accuracy)
- **Range**: -∞ to 1.0 (100%)
- **Interpretation**: Overall tracking quality accounting for false positives, false negatives, and ID switches
- **Higher is better**
- **Formula**: 1 - (FP + FN + ID_switches) / GT_instances

### IDF1 (ID F1 Score)
- **Range**: 0.0 to 1.0 (100%)
- **Interpretation**: Measures how well IDs are maintained over time
- **Higher is better**
- Focus on ID consistency rather than detection quality

### MOTP (Multiple Object Tracking Precision)
- **Range**: 0.0 to 1.0
- **Interpretation**: Average IoU for correctly matched instances
- **Higher is better**
- Measures localization accuracy

### ID Switches
- **Range**: 0 to N (count)
- **Interpretation**: Number of times a tracked instance switches to a different ground truth ID
- **Lower is better**
- Directly measures tracking consistency

### Precision & Recall
- Standard detection metrics
- **Precision**: Of all predictions, how many were correct?
- **Recall**: Of all ground truth bees, how many were detected?

## Tips for Best Results

### Annotation Quality

1. **Consistent IDs**: Ensure the same bee has the same ID across frames
2. **Complete coverage**: Annotate ALL bees in both frames
3. **Diverse scenarios**: Include different activity levels, lighting, occlusions
4. **Multiple videos**: Test across different dates/colonies if possible

### Sequence Selection

1. **Start small**: Begin with 2-frame sequences (easier to annotate)
2. **Representative samples**: Choose frames that represent typical conditions
3. **Challenge cases**: Include some difficult scenarios (crowding, occlusions)
4. **Balance**: Aim for 5-10 sequences across different videos

### Parameter Tuning

1. **Start with defaults**: ByteTrack defaults are usually good
2. **Lower IoU thresholds**: If bees move fast between frames
3. **Increase max_frames_lost**: If occlusions are common
4. **Use mask IoU**: More accurate than bbox IoU for segmentation

## Example Workflow

Let's say you want to optimize tracking for a bumblebox video:

1. **Annotate sequences**:
   - Video: `bumblebox-01_2024-09-18`
   - Sequence 1: Frames 100-101 (low activity)
   - Sequence 2: Frames 500-501 (high activity)
   - Sequence 3: Frames 1000-1001 (bees entering)

2. **Create sequences** in the UI for all three pairs

3. **Run baseline validation**:
   - Test: ByteTrack (default), Simple IoU, Centroid
   - Model: Your best YOLO checkpoint
   - Review results

4. **Optimize ByteTrack** (if it performed best):
   - Adjust `iou_threshold_high` from 0.6 to 0.5 (if bees move fast)
   - Increase `max_frames_lost` from 10 to 20 (if occlusions common)
   - Re-run validation with new parameters

5. **Compare results** and select best configuration

6. **Use the optimized parameters** in your production tracking

## File Structure

After running validation, results are saved to:

```
<project_path>/
  tracking_validation/
    20260405_143022/                    # Timestamp of validation run
      config.json                       # Configuration used
      summary.json                      # Overall metrics summary
      bytetrack_results.csv             # Detailed ByteTrack results
      simple_iou_results.csv            # Detailed Simple IoU results
      centroid_results.csv              # Detailed Centroid results
      comparison.png                    # Visual comparison plot
```

## Troubleshooting

### "No tracking sequences found"
- You need to create tracking sequences first using the Tracking Sequences panel

### "Frame X missing annotations"
- Annotate the missing frame or adjust your sequence range

### "Poor tracking performance"
- Try lowering IoU thresholds
- Ensure ground truth IDs are consistent
- Check if detection quality is good (low confidence detections hurt tracking)

### "ID switches too high"
- Bees may be moving too fast between frames (try lower IoU threshold)
- Ground truth IDs might be inconsistent (review annotations)
- Consider using mask IoU instead of bbox IoU

## Integration with Existing Features

The tracking validation system works alongside:

- **Frame-level validation**: Tests per-frame detection quality
- **Instance tracking**: The ByteTrack implementation used in production
- **YOLO training**: Use validation results to improve your model

## Future Enhancements

Potential additions:

- Longer sequences (3+ frames) for extended tracking
- Re-identification features for handling long occlusions
- Visualization of tracking paths overlaid on videos
- Export of optimized tracking configs for production use

## Support

For issues or questions:
1. Check that annotations exist for all frames in your sequences
2. Review the validation log for specific errors
3. Try with a simpler sequence first (2 frames, fewer bees)
