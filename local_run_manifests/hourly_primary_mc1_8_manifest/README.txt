Hourly primary manifest for MCs 1-8

Created: 2026-07-05
Archive root: /media/august/AutoPolls9/dose-curve-pesticide-data
Symlink folder for GUI input: /tmp/bumblebox_hourly_primary_mc1_8_explicit_links

Selection rule:
- One video per target hour.
- Prefer HH:00.
- If HH:00 is missing but HH:30 exists, use HH:30 and mark selection_status=fallback_hh30_missing_hh00.
- Do not infer microcolony identity from bumblebox filename labels. MC assignment comes from the source folder names.
- Folder period is authoritative for this copy: pre-exposure/ is pre-dose and post-exposure/ is post-dose.
- MCs-1-and-2 pre-exposure includes same-day pre-dose files through 08:50 on 2026-02-23, per the extended pre-day rule.

Primary hourly set:
- 147 videos.
- Use hourly_primary_manifest.csv for metadata.
- Use hourly_primary_file_list.txt or point the GUI at the symlink folder if processing via folder mode.
- For balanced interruptible processing, use hourly_balanced_round_robin_source_file_list.txt in selected-file mode and preserve selected file/list order.
- Balanced order is: for each hourly slot, walk across MC pairs and emit pre then post videos when present.
- For per-MC-pair ArUco filtering, use mc1_8_tag_allow_map.csv in the GUI/CLI allow tags field.
- The tag map was generated from /home/august/Documents/Microcolony Metadata.xlsx.

Supplemental partial set:
- 0 videos. All intended pre/post files are represented by pre-exposure/ and post-exposure/.

Suggested output folder for first analysis:
/media/august/AutoPolls9/dose-curve-pesticide-data/tracking_hourly_balanced_mc1_8
