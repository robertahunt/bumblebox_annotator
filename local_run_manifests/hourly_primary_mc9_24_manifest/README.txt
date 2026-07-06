Hourly primary manifest for MCs 9-24

Created: 2026-07-03
Archive root: /media/august/Expansion/Behavioral-Entropy-Pesticide-Dosing-Exps
Symlink folder for GUI input: /tmp/bumblebox_hourly_primary_mc9_24_links

Selection rule:
- One video per target hour.
- Prefer HH:00.
- If HH:00 is missing but HH:30 exists, use HH:30 and mark selection_status=fallback_hh30_missing_hh00.
- Do not infer microcolony identity from bumblebox filename labels. MC assignment comes from the destination/source-copy folder names.

Primary hourly set:
- 333 videos.
- Use hourly_primary_manifest.csv for metadata.
- Use hourly_primary_file_list.txt or point the GUI at the symlink folder if processing via folder mode.
- For balanced interruptible processing, use hourly_balanced_round_robin_source_file_list.txt in selected-file mode and keep "Preserve selected file/list order" checked.
- Balanced order is: for each hourly slot, walk across MC pairs and emit pre-dose then post-dose videos when present.
- For per-MC-pair ArUco filtering, use mc9_24_tag_allow_map.csv in the GUI "Allow tags" field.
- The tag map was generated from /home/august/Documents/Microcolony Metadata.xlsx; MCs-15-and-16 also includes manually added tag 29.

Supplemental partial set:
- 2 videos, stored in hourly_supplemental_partial_manifest.csv.
- MCs-21-and-22 original D+3 day (2026-03-19) was only the first few hours, so it is excluded from the primary hourly symlink folder.
- MCs-21-and-22 primary post-dose substitute is 2026-03-18 (D+2).

Suggested output folder for first analysis:
/media/august/Expansion/Behavioral-Entropy-Pesticide-Dosing-Exps/tracking_hourly_balanced_mc9_24
