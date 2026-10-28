AWE Q20 orbit 4 v02, exact Epoch match to previous demonstration.
Source: E:\soc\l1r\2023\11\awe_l1r_q20_2023326T0542_00004_v02.nc
49 exact matches to earlier v01 indices 930-979. v01 frame 958 has no exact v02 Epoch match and is omitted.
No replacement, duplicate, interpolation in time, or synthetic frame is inserted.
All 49 matches have exactly equal float64 Epoch values. See epoch_mapping.csv.
v02 source indices are 1154-1202. Reference v01 frame 955 corresponds to v02 index 1179.
v02 geolocation assumes 85 km, not the earlier 87 km. The same geographic output grid is used.
Missing radiance vertices are retained during spatial interpolation to preserve v02 masks.
Extract the entire ZIP. Open AWE_v02_actual_data.ipynb in Jupyter and run cells in order.
Set ITERATIONS to 256 or another count. Default is 64.
Default velocities have been freshly estimated from these v02 frames, not reused from v01.
The optional original-frame section reads the supplied subset; no E: drive is required locally.
Original global date metadata refer to the full source file; Epoch and mapping specify subset times.
Reconstruction residual is not ground-truth separation error. More iterations need not yield pure layers.
