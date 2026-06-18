#!/usr/bin/env python3
"""
Identify which ArUco dictionary type is used in a folder of example images.

Usage:
    python identify_aruco_dict.py <folder_path>
    
Example:
    python identify_aruco_dict.py ./aruco_examples/
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict


# All available ArUco dictionaries
ARUCO_DICTS = {
    '4x4_50': cv2.aruco.DICT_4X4_50,
    '4x4_100': cv2.aruco.DICT_4X4_100,
    '4x4_250': cv2.aruco.DICT_4X4_250,
    '4x4_1000': cv2.aruco.DICT_4X4_1000,
    '5x5_50': cv2.aruco.DICT_5X5_50,
    '5x5_100': cv2.aruco.DICT_5X5_100,
    '5x5_250': cv2.aruco.DICT_5X5_250,
    '5x5_1000': cv2.aruco.DICT_5X5_1000,
    '6x6_50': cv2.aruco.DICT_6X6_50,
    '6x6_100': cv2.aruco.DICT_6X6_100,
    '6x6_250': cv2.aruco.DICT_6X6_250,
    '6x6_1000': cv2.aruco.DICT_6X6_1000,
    '7x7_50': cv2.aruco.DICT_7X7_50,
    '7x7_100': cv2.aruco.DICT_7X7_100,
    '7x7_250': cv2.aruco.DICT_7X7_250,
    '7x7_1000': cv2.aruco.DICT_7X7_1000,
    'ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    'APRILTAG_16h5': cv2.aruco.DICT_APRILTAG_16h5,
    'APRILTAG_25h9': cv2.aruco.DICT_APRILTAG_25h9,
    'APRILTAG_36h10': cv2.aruco.DICT_APRILTAG_36h10,
    'APRILTAG_36h11': cv2.aruco.DICT_APRILTAG_36h11,
}


def detect_with_dict(image, dict_name, dict_type):
    """Try to detect ArUco markers with a specific dictionary
    
    Returns:
        List of (marker_id, corners) tuples if detected, empty list otherwise
    """
    # Create detector for this dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    detector_params = cv2.aruco.DetectorParameters()
    
    # Adjust parameters for better detection
    detector_params.adaptiveThreshConstant = 7
    detector_params.minMarkerPerimeterRate = 0.01  # Lower threshold for small markers
    detector_params.maxMarkerPerimeterRate = 10.0
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)
    
    if ids is not None and len(ids) > 0:
        detections = []
        for i, marker_id in enumerate(ids):
            detections.append((int(marker_id[0]), corners[i]))
        return detections
    
    return []


def analyze_image(image_path):
    """Analyze a single image and try all ArUco dictionaries
    
    Returns:
        Dict mapping dict_name -> list of detected marker IDs
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  WARNING: Could not load {image_path}")
        return {}
    
    # Convert to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = {}
    
    # Try each dictionary
    for dict_name, dict_type in ARUCO_DICTS.items():
        detections = detect_with_dict(gray, dict_name, dict_type)
        if detections:
            marker_ids = [marker_id for marker_id, _ in detections]
            results[dict_name] = marker_ids
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python identify_aruco_dict.py <folder_path>")
        print("\nExample: python identify_aruco_dict.py ./aruco_examples/")
        sys.exit(1)
    
    folder_path = Path(sys.argv[1])
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"ERROR: Not a directory: {folder_path}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"ERROR: No image files found in {folder_path}")
        print(f"Looked for: {', '.join(image_extensions)}")
        sys.exit(1)
    
    print("="*80)
    print(f"ARUCO DICTIONARY IDENTIFIER")
    print("="*80)
    print(f"\nAnalyzing {len(image_files)} image(s) in: {folder_path}\n")
    
    # Track which dictionaries work across all images
    dict_counts = defaultdict(int)  # How many images each dict successfully detected
    dict_marker_ids = defaultdict(set)  # Which marker IDs each dict found
    total_images = len(image_files)
    
    # Analyze each image
    for image_path in sorted(image_files):
        print(f"\n📷 {image_path.name}")
        print("-" * 80)
        
        results = analyze_image(image_path)
        
        if not results:
            print("  ❌ No ArUco markers detected with any dictionary")
        else:
            for dict_name, marker_ids in sorted(results.items()):
                print(f"  ✓ {dict_name:20s} detected marker ID(s): {marker_ids}")
                dict_counts[dict_name] += 1
                dict_marker_ids[dict_name].update(marker_ids)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not dict_counts:
        print("\n❌ No ArUco markers were detected in any images with any dictionary.")
        print("\nPossible reasons:")
        print("  - Images don't contain ArUco markers")
        print("  - Markers are too small or distorted")
        print("  - Markers are custom/non-standard")
        print("  - Images are corrupted or unreadable")
    else:
        # Sort by detection count (descending)
        sorted_dicts = sorted(dict_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n✓ Detected markers in {len(image_files)} image(s)\n")
        print("Dictionary Detection Results:")
        print(f"{'Dictionary':<20s} {'Images':<10s} {'Success Rate':<15s} {'Marker IDs'}")
        print("-" * 80)
        
        for dict_name, count in sorted_dicts:
            success_rate = (count / total_images) * 100
            marker_ids = sorted(dict_marker_ids[dict_name])
            marker_ids_str = str(marker_ids) if len(marker_ids) <= 10 else f"{marker_ids[:10]} + {len(marker_ids)-10} more"
            print(f"{dict_name:<20s} {count:>3d}/{total_images:<5d} {success_rate:>5.1f}%         {marker_ids_str}")
        
        # Recommend best dictionary
        best_dict, best_count = sorted_dicts[0]
        if best_count == total_images:
            print(f"\n🎯 RECOMMENDATION: Use '{best_dict}' (detected in all images)")
        else:
            print(f"\n⚠️  RECOMMENDATION: Use '{best_dict}' (detected in most images)")
            print(f"   Note: Some images may contain markers from different dictionaries")
        
        # Check for multiple dictionaries with same detection rate
        top_dicts = [d for d, c in sorted_dicts if c == best_count]
        if len(top_dicts) > 1:
            print(f"\n⚠️  WARNING: Multiple dictionaries detected the same markers:")
            for dict_name in top_dicts:
                ids = sorted(dict_marker_ids[dict_name])
                print(f"   - {dict_name}: IDs {ids}")
            print(f"   This means the marker IDs overlap between dictionaries.")
            print(f"   Choose the dictionary that matches your marker source.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
