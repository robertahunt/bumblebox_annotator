"""
Test script to compare annotation save/load performance:
- Current approach: pickle files with RLE-compressed masks
- Alternative approach: PNG files for masks + JSON for metadata

This script simulates realistic bee annotation data and measures:
1. Save time
2. Load time
3. File size
4. Overall round-trip time
"""

import numpy as np
import pickle
import json
import time
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import cv2


# Copy the RLE functions from annotation.py (original implementation)
def mask_to_rle(mask):
    """Convert binary mask to run-length encoding (RLE)"""
    pixels = (mask.flatten() > 0).astype(np.uint8)
    
    if len(pixels) == 0:
        return {'size': list(mask.shape), 'counts': []}
    
    counts = []
    current_value = 0
    current_count = 0
    
    for pixel in pixels:
        if pixel == current_value:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1
            current_value = 1 - current_value
    
    counts.append(current_count)
    
    if current_value == 0:
        counts.append(0)
    
    return {
        'size': list(mask.shape),
        'counts': counts
    }


def rle_to_mask(rle):
    """Convert run-length encoding to binary mask"""
    h, w = rle['size']
    counts = rle['counts']
    
    if len(counts) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    total_pixels = h * w
    mask = np.zeros(total_pixels, dtype=np.uint8)
    
    position = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            end_pos = min(position + count, total_pixels)
            mask[position:end_pos] = 255
        position += count
        if position >= total_pixels:
            break
    
    return mask.reshape((h, w))


def generate_bee_mask(image_size, center, size_range=(30, 80)):
    """Generate a realistic bee-shaped mask (ellipse)"""
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y = center
    width = np.random.randint(*size_range)
    height = int(width * np.random.uniform(0.6, 0.9))  # Bees are roughly oval
    angle = np.random.randint(0, 180)
    
    cv2.ellipse(mask, (x, y), (width // 2, height // 2), angle, 0, 360, 255, -1)
    return mask


def generate_sample_annotations(num_annotations=20, image_size=(1080, 1920)):
    """Generate sample annotations similar to bee annotation data"""
    annotations = []
    
    for i in range(num_annotations):
        # Random position for bee
        x = np.random.randint(50, image_size[1] - 50)
        y = np.random.randint(50, image_size[0] - 50)
        
        # Generate mask
        mask = generate_bee_mask(image_size, (x, y))
        
        # Calculate bounding box
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        else:
            bbox = [x - 20, y - 20, 40, 40]
        
        annotation = {
            'class': 'bee',
            'class_id': 0,
            'bbox': bbox,
            'confidence': np.random.uniform(0.8, 1.0),
            'mask': mask
        }
        annotations.append(annotation)
    
    return annotations


# ============================================================================
# METHOD 1: Current approach (Pickle + RLE)
# ============================================================================

def save_annotations_pickle_rle(annotations, filepath):
    """Current method: Save annotations as pickle with RLE-compressed masks"""
    compressed_annotations = []
    for ann in annotations:
        compressed_ann = ann.copy()
        if 'mask' in ann:
            compressed_ann['mask_rle'] = mask_to_rle(ann['mask'])
            del compressed_ann['mask']
        compressed_annotations.append(compressed_ann)
    
    with open(filepath, 'wb') as f:
        pickle.dump(compressed_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_annotations_pickle_rle(filepath):
    """Current method: Load annotations from pickle and decompress RLE masks"""
    with open(filepath, 'rb') as f:
        compressed_annotations = pickle.load(f)
    
    annotations = []
    for ann in compressed_annotations:
        decompressed_ann = ann.copy()
        if 'mask_rle' in ann:
            decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
            del decompressed_ann['mask_rle']
        annotations.append(decompressed_ann)
    
    return annotations


# ============================================================================
# METHOD 2: PNG approach
# ============================================================================

def save_annotations_png(annotations, base_path):
    """Alternative: Save masks as PNG files and metadata as JSON"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    
    for i, ann in enumerate(annotations):
        # Save mask as PNG
        mask_filename = f"mask_{i:04d}.png"
        mask_path = base_path / mask_filename
        
        if 'mask' in ann:
            # Save as grayscale PNG (values 0 or 255)
            Image.fromarray(ann['mask']).save(mask_path, 'PNG', compress_level=6)
        
        # Save metadata (everything except mask)
        metadata = {k: v for k, v in ann.items() if k != 'mask'}
        metadata['mask_file'] = mask_filename
        metadata_list.append(metadata)
    
    # Save all metadata as JSON
    json_path = base_path / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata_list, f)


def load_annotations_png(base_path):
    """Alternative: Load masks from PNG files and metadata from JSON"""
    base_path = Path(base_path)
    
    # Load metadata
    json_path = base_path / 'metadata.json'
    with open(json_path, 'r') as f:
        metadata_list = json.load(f)
    
    annotations = []
    for metadata in metadata_list:
        ann = metadata.copy()
        
        # Load mask from PNG
        mask_filename = ann.pop('mask_file')
        mask_path = base_path / mask_filename
        
        mask = np.array(Image.open(mask_path))
        ann['mask'] = mask
        
        annotations.append(ann)
    
    return annotations


# ============================================================================
# Performance Testing
# ============================================================================

def get_directory_size(path):
    """Calculate total size of a directory in bytes"""
    total_size = 0
    for file in Path(path).rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size


def test_method(method_name, save_func, load_func, annotations, test_dir, num_iterations=10):
    """Test a save/load method and return performance metrics"""
    print(f"\n{'='*70}")
    print(f"Testing: {method_name}")
    print(f"{'='*70}")
    
    save_times = []
    load_times = []
    
    for i in range(num_iterations):
        # Test save
        start = time.time()
        save_func(annotations, test_dir)
        save_time = time.time() - start
        save_times.append(save_time)
        
        # Test load
        start = time.time()
        loaded_annotations = load_func(test_dir)
        load_time = time.time() - start
        load_times.append(load_time)
        
        # Verify correctness (only on first iteration)
        if i == 0:
            assert len(loaded_annotations) == len(annotations), "Annotation count mismatch!"
            for orig, loaded in zip(annotations, loaded_annotations):
                assert np.array_equal(orig['mask'], loaded['mask']), "Mask mismatch!"
            print("✓ Data integrity verified")
    
    # Calculate statistics
    avg_save = np.mean(save_times)
    std_save = np.std(save_times)
    avg_load = np.mean(load_times)
    std_load = np.std(load_times)
    avg_total = avg_save + avg_load
    
    # Get file size
    file_size = get_directory_size(test_dir)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\nResults over {num_iterations} iterations:")
    print(f"  Save time:  {avg_save*1000:.2f} ± {std_save*1000:.2f} ms")
    print(f"  Load time:  {avg_load*1000:.2f} ± {std_load*1000:.2f} ms")
    print(f"  Total time: {avg_total*1000:.2f} ms")
    print(f"  File size:  {file_size_mb:.3f} MB ({file_size:,} bytes)")
    
    return {
        'method': method_name,
        'avg_save_ms': avg_save * 1000,
        'std_save_ms': std_save * 1000,
        'avg_load_ms': avg_load * 1000,
        'std_load_ms': std_load * 1000,
        'avg_total_ms': avg_total * 1000,
        'file_size_bytes': file_size,
        'file_size_mb': file_size_mb
    }


def main():
    """Run comprehensive performance comparison"""
    print("="*70)
    print("ANNOTATION SAVE/LOAD PERFORMANCE TEST")
    print("="*70)
    
    # Test configurations (reduced for faster testing)
    test_configs = [
        {'num_annotations': 10, 'image_size': (1080, 1920)},
        {'num_annotations': 20, 'image_size': (1080, 1920)},
    ]
    
    all_results = []
    
    for config in test_configs:
        num_annotations = config['num_annotations']
        image_size = config['image_size']
        
        print(f"\n\n{'#'*70}")
        print(f"# TEST CONFIGURATION: {num_annotations} annotations, {image_size[0]}x{image_size[1]} image")
        print(f"{'#'*70}")
        
        # Generate sample data
        print(f"\nGenerating {num_annotations} sample annotations...")
        annotations = generate_sample_annotations(num_annotations, image_size)
        print(f"✓ Generated {len(annotations)} annotations")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_base:
            temp_base = Path(temp_base)
            
            # Test Method 1: Pickle + RLE
            pickle_dir = temp_base / 'pickle_test'
            pickle_file = pickle_dir / 'frame_000000.pkl'
            pickle_dir.mkdir()
            
            result1 = test_method(
                "Pickle + RLE (Current Method)",
                lambda ann, path: save_annotations_pickle_rle(ann, pickle_file),
                lambda path: load_annotations_pickle_rle(pickle_file),
                annotations,
                pickle_dir,
                num_iterations=3
            )
            result1['num_annotations'] = num_annotations
            all_results.append(result1)
            
            # Clean up for next test
            shutil.rmtree(pickle_dir)
            
            # Test Method 2: PNG
            png_dir = temp_base / 'png_test'
            
            result2 = test_method(
                "PNG + JSON (Alternative Method)",
                save_annotations_png,
                load_annotations_png,
                annotations,
                png_dir,
                num_iterations=3
            )
            result2['num_annotations'] = num_annotations
            all_results.append(result2)
    
    # Print comparison summary
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    for i in range(0, len(all_results), 2):
        pickle_result = all_results[i]
        png_result = all_results[i + 1]
        
        print(f"\n{pickle_result['num_annotations']} Annotations:")
        print(f"{'Metric':<25} | {'Pickle+RLE':<15} | {'PNG+JSON':<15} | {'Winner':<20}")
        print("-" * 70)
        
        # Save time
        speedup = pickle_result['avg_save_ms'] / png_result['avg_save_ms']
        winner = "PNG+JSON" if speedup > 1 else "Pickle+RLE"
        print(f"{'Save time':<25} | {pickle_result['avg_save_ms']:>13.2f} | {png_result['avg_save_ms']:>13.2f} | {winner:<15} ({abs(speedup-1)*100:>4.1f}%)")
        
        # Load time
        speedup = pickle_result['avg_load_ms'] / png_result['avg_load_ms']
        winner = "PNG+JSON" if speedup > 1 else "Pickle+RLE"
        print(f"{'Load time':<25} | {pickle_result['avg_load_ms']:>13.2f} | {png_result['avg_load_ms']:>13.2f} | {winner:<15} ({abs(speedup-1)*100:>4.1f}%)")
        
        # Total time
        speedup = pickle_result['avg_total_ms'] / png_result['avg_total_ms']
        winner = "PNG+JSON" if speedup > 1 else "Pickle+RLE"
        print(f"{'Round-trip time':<25} | {pickle_result['avg_total_ms']:>13.2f} | {png_result['avg_total_ms']:>13.2f} | {winner:<15} ({abs(speedup-1)*100:>4.1f}%)")
        
        # File size
        compression = pickle_result['file_size_mb'] / png_result['file_size_mb']
        winner = "PNG+JSON" if compression < 1 else "Pickle+RLE"
        print(f"{'File size (MB)':<25} | {pickle_result['file_size_mb']:>13.3f} | {png_result['file_size_mb']:>13.3f} | {winner:<15} ({abs(compression-1)*100:>4.1f}%)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Calculate average performance across all tests
    pickle_results = [r for r in all_results if 'Pickle' in r['method']]
    png_results = [r for r in all_results if 'PNG' in r['method']]
    
    avg_pickle_total = np.mean([r['avg_total_ms'] for r in pickle_results])
    avg_png_total = np.mean([r['avg_total_ms'] for r in png_results])
    
    if avg_png_total < avg_pickle_total:
        speedup = ((avg_pickle_total - avg_png_total) / avg_pickle_total) * 100
        print(f"\n✓ PNG+JSON is FASTER on average by {speedup:.1f}%")
        print(f"  Average round-trip: {avg_png_total:.2f} ms vs {avg_pickle_total:.2f} ms")
    else:
        speedup = ((avg_png_total - avg_pickle_total) / avg_png_total) * 100
        print(f"\n✓ Pickle+RLE is FASTER on average by {speedup:.1f}%")
        print(f"  Average round-trip: {avg_pickle_total:.2f} ms vs {avg_png_total:.2f} ms")
    
    print("\nConsiderations:")
    print("  • PNG files are more human-readable and debuggable")
    print("  • Pickle+RLE stores everything in a single file per frame")
    print("  • PNG approach creates multiple files per frame")
    print("  • Actual performance may vary based on disk I/O and annotation density")


if __name__ == '__main__':
    main()
