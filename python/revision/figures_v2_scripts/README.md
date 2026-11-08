# figures_v2 scripts

- `make_fig01_study_area_v2.py` — initial cascade (uses parsed geojson; may miss POLY-0051+)
- Regenerate with full holdout set: rebuild from `Confirmed_apple_field_points_area.kml` all layers → `user_apple_polygons_full_v2.geojson`, then rerun the inline rebuild in the figpass session (or merge full geojson into the script).
- `make_fig02_framework_v2.py` — five-column framework

Python: `/workspace/.venv_apple/bin/python`
Out: `outputs/revision/figures_v2/`
