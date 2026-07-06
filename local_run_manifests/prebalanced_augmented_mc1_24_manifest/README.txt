Prebalanced augmented manifest for MCs 1-24

Purpose:
- Keep the existing hourly primary post-exposure selections unchanged.
- Add non-hourly pre-exposure videos to bring partial pre coverage toward 24 videos per MC pair when raw videos exist.
- Add extra pre-exposure same-treatment videos to compensate for blurry MC1/2 ArUco tracking limitations.

Files:
- prebalanced_augmented_round_robin_manifest.csv
- prebalanced_augmented_round_robin_source_file_list.txt
- augmentation_log.csv
- treatment_counts_by_round.csv
- treatment_counts_all_rounds.csv

Unique selected pair-video files:
- Pre: 285
- Post: 285
- Total: 570

Notes:
- MC1 and MC2 remain in the manifest, but extra same-treatment pre videos were added from MC14/MC21 (0.5ppb) and MC17 (10ppb) for ArUco compensation.
- MC20 is grouped under 2.5ppb because the workbook has a 2ppb/2.5ppb inconsistency and the design is the 8-condition dose set.
