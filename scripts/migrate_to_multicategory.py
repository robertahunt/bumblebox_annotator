"""
Migration script to add multi-category support to existing annotations.

This script:
1. Backs up existing annotations
2. Adds category_id field to all existing annotations (defaults to 1 for 'bee')
3. Updates project metadata to include new class names
4. Regenerates COCO files with updated category definitions

Usage:
    python -m scripts.migrate_to_multicategory <project_path>
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import sys


def backup_annotations(project_path):
    """Create backup of annotations directory"""
    annotations_dir = project_path / 'annotations'
    if not annotations_dir.exists():
        print(f"No annotations directory found at {annotations_dir}")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = project_path / f'annotations_backup_{timestamp}'
    
    print(f"Creating backup: {backup_dir}")
    shutil.copytree(annotations_dir, backup_dir)
    print(f"✓ Backup created successfully")
    
    return backup_dir


def migrate_json_annotations(project_path):
    """Add category_id to all JSON annotation files"""
    json_dir = project_path / 'annotations' / 'json'
    
    if not json_dir.exists():
        print(f"No JSON annotations directory found at {json_dir}")
        return 0, 0
    
    updated_files = 0
    updated_annotations = 0
    skipped_files = 0
    
    # Process each video directory
    for video_dir in json_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        print(f"Processing video: {video_dir.name}")
        
        # Process each frame annotation file
        for json_file in video_dir.glob('frame_*.json'):
            try:
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
                
                # Check if any annotation needs migration
                needs_update = False
                for ann in annotations:
                    if 'category_id' not in ann:
                        needs_update = True
                        ann['category_id'] = 1  # Default to 'bee'
                        updated_annotations += 1
                
                # Write back if changed
                if needs_update:
                    with open(json_file, 'w') as f:
                        json.dump(annotations, f, indent=2)
                    updated_files += 1
                else:
                    skipped_files += 1
                    
            except Exception as e:
                print(f"  ERROR processing {json_file}: {e}")
    
    return updated_files, updated_annotations


def migrate_bbox_annotations(project_path):
    """Add category_id to all bbox annotation files"""
    bbox_dir = project_path / 'annotations' / 'bbox'
    
    if not bbox_dir.exists():
        print(f"No bbox annotations directory found at {bbox_dir}")
        return 0, 0
    
    updated_files = 0
    updated_annotations = 0
    
    # Process each video directory
    for video_dir in bbox_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        print(f"Processing bbox video: {video_dir.name}")
        
        # Process each frame annotation file
        for json_file in video_dir.glob('frame_*.json'):
            try:
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
                
                # Check if any annotation needs migration
                needs_update = False
                for ann in annotations:
                    if 'category_id' not in ann:
                        needs_update = True
                        ann['category_id'] = 1  # Default to 'bee'
                        updated_annotations += 1
                
                # Write back if changed
                if needs_update:
                    with open(json_file, 'w') as f:
                        json.dump(annotations, f, indent=2)
                    updated_files += 1
                    
            except Exception as e:
                print(f"  ERROR processing {json_file}: {e}")
    
    return updated_files, updated_annotations


def update_project_metadata(project_path):
    """Update project metadata to include new class names"""
    project_file = project_path / 'annotations' / 'project.json'
    
    if not project_file.exists():
        print(f"No project file found at {project_file}")
        return False
    
    try:
        with open(project_file, 'r') as f:
            project_data = json.load(f)
        
        # Update class names if needed
        current_classes = project_data.get('classes', ['bee'])
        new_classes = ['bee', 'hive', 'chamber']
        
        if current_classes != new_classes:
            project_data['classes'] = new_classes
            project_data['modified'] = datetime.now().isoformat()
            project_data['migration_date'] = datetime.now().isoformat()
            project_data['migration_version'] = '2.1'
            
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            print(f"✓ Updated project metadata: {current_classes} -> {new_classes}")
            return True
        else:
            print(f"Project metadata already up to date")
            return False
            
    except Exception as e:
        print(f"ERROR updating project metadata: {e}")
        return False


def regenerate_coco(project_path):
    """Regenerate COCO files with updated categories"""
    # Import here to avoid circular dependencies
    try:
        from training.coco_video_export import export_coco_per_video
    except ImportError:
        print("WARNING: Could not import COCO export function. Skipping COCO regeneration.")
        return False
    
    try:
        from core.project_manager import ProjectManager
        
        pm = ProjectManager(project_path)
        pm.load_project(project_path)
        
        train_videos = pm.scan_videos().get('train', [])
        val_videos = pm.scan_videos().get('val', [])
        
        class_names = ['bee', 'hive', 'chamber']
        
        print("\nRegenerating COCO datasets...")
        
        if train_videos:
            print(f"  Exporting training set ({len(train_videos)} videos)...")
            train_paths = export_coco_per_video(
                project_path,
                train_videos,
                'train',
                class_names=class_names
            )
            print(f"  ✓ Training: {len(train_paths)} files")
        
        if val_videos:
            print(f"  Exporting validation set ({len(val_videos)} videos)...")
            val_paths = export_coco_per_video(
                project_path,
                val_videos,
                'val',
                class_names=class_names
            )
            print(f"  ✓ Validation: {len(val_paths)} files")
        
        print("✓ COCO datasets regenerated")
        return True
        
    except Exception as e:
        print(f"ERROR regenerating COCO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Migrate annotations to multi-category format'
    )
    parser.add_argument(
        'project_path',
        type=str,
        help='Path to project directory'
    )
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip creating backup (not recommended)'
    )
    parser.add_argument(
        '--skip-coco',
        action='store_true',
        help='Skip COCO regeneration'
    )
    
    args = parser.parse_args()
    project_path = Path(args.project_path)
    
    if not project_path.exists():
        print(f"ERROR: Project path does not exist: {project_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Multi-Category Annotation Migration")
    print("=" * 70)
    print(f"Project: {project_path}")
    print()
    
    # Step 1: Backup
    if not args.skip_backup:
        backup_dir = backup_annotations(project_path)
        if backup_dir:
            print(f"Backup location: {backup_dir}")
        print()
    else:
        print("⚠ WARNING: Skipping backup (--skip-backup flag used)")
        print()
    
    # Step 2: Migrate JSON annotations
    print("Step 1: Migrating JSON annotations...")
    json_files, json_anns = migrate_json_annotations(project_path)
    print(f"✓ Updated {json_files} JSON files ({json_anns} annotations)")
    print()
    
    # Step 3: Migrate bbox annotations
    print("Step 2: Migrating bbox annotations...")
    bbox_files, bbox_anns = migrate_bbox_annotations(project_path)
    print(f"✓ Updated {bbox_files} bbox files ({bbox_anns} annotations)")
    print()
    
    # Step 4: Update project metadata
    print("Step 3: Updating project metadata...")
    update_project_metadata(project_path)
    print()
    
    # Step 5: Regenerate COCO
    if not args.skip_coco:
        print("Step 4: Regenerating COCO datasets...")
        regenerate_coco(project_path)
        print()
    else:
        print("⚠ Skipping COCO regeneration (--skip-coco flag used)")
        print()
    
    print("=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    print(f"Total annotations updated: {json_anns + bbox_anns}")
    print(f"Total files updated: {json_files + bbox_files}")
    print()
    print("All existing annotations have been assigned category_id=1 (bee)")
    print("You can now annotate hives (category_id=2) and chambers (category_id=3)")
    print()


if __name__ == '__main__':
    main()
