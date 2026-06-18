"""
Add timing instrumentation to main_window.py to diagnose frame loading bottlenecks

This script adds timing measurements to the load_frame method to identify
which operations are slow.

Usage:
1. Backup your main_window.py first!
2. Run this script to add timing code
3. Run your app and navigate between frames
4. Check the console output for timing breakdowns
5. Run this script again with --remove to restore original code
"""

from pathlib import Path
import sys

MAIN_WINDOW_PATH = Path(__file__).parent / "gui" / "main_window.py"

# Timing code to insert at the start of load_frame method
TIMING_START = """
        # ==== PERFORMANCE PROFILING START ====
        import time
        _timing = {}
        _timing['start'] = time.perf_counter()
"""

# Timing code to insert after image loading
TIMING_AFTER_IMAGE = """
        _timing['after_image'] = time.perf_counter()
"""

# Timing code to insert after annotation loading
TIMING_AFTER_LOAD_ANN = """
                    _timing['after_load_ann'] = time.perf_counter()
"""

# Timing code to insert after contour cleaning
TIMING_AFTER_CLEAN = """
                    _timing['after_clean'] = time.perf_counter()
"""

# Timing code to insert after set_annotations
TIMING_AFTER_SET_ANN = """
                _timing['after_set_ann'] = time.perf_counter()
"""

# Timing code to insert at the end of load_frame (before the except block)
TIMING_END = """
                # ==== PERFORMANCE PROFILING END ====
                if '_timing' in locals():
                    t = _timing
                    total = (t.get('after_set_ann', t['start']) - t['start']) * 1000
                    image = (t.get('after_image', t['start']) - t['start']) * 1000
                    load_ann = (t.get('after_load_ann', t.get('after_image', t['start'])) - t.get('after_image', t['start'])) * 1000
                    clean = (t.get('after_clean', t.get('after_load_ann', t.get('after_image', t['start']))) - t.get('after_load_ann', t.get('after_image', t['start']))) * 1000
                    set_ann = (t.get('after_set_ann', t.get('after_clean', t.get('after_load_ann', t.get('after_image', t['start'])))) - t.get('after_clean', t.get('after_load_ann', t.get('after_image', t['start'])))) * 1000
                    
                    print(f"\\n{'='*80}")
                    print(f"FRAME {idx} LOADING PERFORMANCE:")
                    print(f"{'='*80}")
                    print(f"  Total:              {total:>7.1f} ms")
                    print(f"  ├─ Image loading:   {image:>7.1f} ms ({image/total*100:>5.1f}%)")
                    print(f"  ├─ Load annot:      {load_ann:>7.1f} ms ({load_ann/total*100:>5.1f}%)")
                    print(f"  ├─ Clean contours:  {clean:>7.1f} ms ({clean/total*100:>5.1f}%)")
                    print(f"  └─ Set annotations: {set_ann:>7.1f} ms ({set_ann/total*100:>5.1f}%)")
                    print(f"{'='*80}\\n")
"""


def add_timing():
    """Add timing instrumentation to load_frame method"""
    
    if not MAIN_WINDOW_PATH.exists():
        print(f"Error: {MAIN_WINDOW_PATH} not found")
        return False
    
    # Read the file
    with open(MAIN_WINDOW_PATH, 'r') as f:
        content = f.read()
    
    # Check if already instrumented
    if "PERFORMANCE PROFILING START" in content:
        print("Timing code already present. Use --remove to restore original.")
        return False
    
    # Find insertion points and add markers
    lines = content.split('\n')
    new_lines = []
    
    in_load_frame = False
    added_start = False
    added_after_image = False
    added_after_load = False
    added_after_clean = False
    added_after_set = False
    added_end = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Detect load_frame method
        if 'def load_frame(self, idx)' in line:
            in_load_frame = True
            print("Found load_frame method")
        
        # Add timing start after the method docstring
        if in_load_frame and not added_start and 'if 0 <= idx < len(self.frames):' in line:
            new_lines.extend(TIMING_START.rstrip('\n').split('\n'))
            added_start = True
            print("  ✓ Added timing start")
        
        # Add timing after image is loaded
        if in_load_frame and not added_after_image and 'image_to_load = frame' in line:
            # Look ahead for the try: block
            if i + 1 < len(lines) and 'try:' in lines[i + 1]:
                new_lines.append(lines[i + 1])  # Add the 'try:' line
                i += 1
                new_lines.extend(TIMING_AFTER_IMAGE.rstrip('\n').split('\n'))
                added_after_image = True
                print("  ✓ Added timing after image load")
        
        # Add timing after annotation loading
        if in_load_frame and not added_after_load and 'self.annotation_manager.load_frame_annotations(' in line:
            # Find the end of this statement
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                new_lines.append(lines[j])
                j += 1
            if j < len(lines):
                new_lines.append(lines[j])  # Add the closing line
                i = j
            new_lines.extend(TIMING_AFTER_LOAD_ANN.rstrip('\n').split('\n'))
            added_after_load = True
            print("  ✓ Added timing after load annotations")
        
        # Add timing after contour cleaning
        if in_load_frame and not added_after_clean and '_clean_duplicate_contours' in line:
            # This is within an if block, so just add after this line
            new_lines.extend(TIMING_AFTER_CLEAN.rstrip('\n').split('\n'))
            added_after_clean = True
            print("  ✓ Added timing after clean contours")
        
        # Add timing after set_annotations
        if in_load_frame and not added_after_set and 'self.canvas.set_annotations(annotations' in line:
            new_lines.extend(TIMING_AFTER_SET_ANN.rstrip('\n').split('\n'))
            added_after_set = True
            print("  ✓ Added timing after set_annotations")
        
        # Add summary at the end (before the except block)
        if in_load_frame and not added_end and 'except Exception as e:' in line:
            new_lines.extend(TIMING_END.rstrip('\n').split('\n'))
            added_end = True
            print("  ✓ Added timing summary")
            in_load_frame = False  # Exit load_frame context
        
        i += 1
    
    # Write back
    with open(MAIN_WINDOW_PATH, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"\n✓ Timing instrumentation added to {MAIN_WINDOW_PATH}")
    print("Run your app and navigate between frames to see performance breakdown.")
    return True


def remove_timing():
    """Remove timing instrumentation"""
    
    if not MAIN_WINDOW_PATH.exists():
        print(f"Error: {MAIN_WINDOW_PATH} not found")
        return False
    
    with open(MAIN_WINDOW_PATH, 'r') as f:
        content = f.read()
    
    # Check if instrumented
    if "PERFORMANCE PROFILING START" not in content:
        print("No timing code found to remove.")
        return False
    
    # Remove all timing code
    lines = content.split('\n')
    new_lines = []
    skip_mode = False
    
    for line in lines:
        if "==== PERFORMANCE PROFILING" in line:
            skip_mode = True
            continue
        if skip_mode:
            # Check if this is actual timing code
            if "_timing" in line or "FRAME" in line and "LOADING PERFORMANCE" in line or "='*80" in line:
                continue
            else:
                skip_mode = False
        
        if not skip_mode:
            new_lines.append(line)
    
    with open(MAIN_WINDOW_PATH, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"✓ Timing instrumentation removed from {MAIN_WINDOW_PATH}")
    return True


if __name__ == '__main__':
    if '--remove' in sys.argv:
        remove_timing()
    else:
        add_timing()
