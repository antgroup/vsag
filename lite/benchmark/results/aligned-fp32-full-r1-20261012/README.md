# Rejected Lite Ofast alignment attempt

Do not use this directory as a performance result. Original Lite was rebuilt in
isolated RAM with Ofast/g to match the earlier native Full release flags.
CTest failed3/5 groups: lite_unit, lite_graph, lite_rabitq_api. NaN/Inf rejection,
prepared-state exceptions, distance/path parity and Batch4/scalar bit parity
failed. Logs retained; measurement stopped before candidate/fresh-query phases.
Production code/flags remained unchanged. The replacement comparison instead
builds isolated native Full with O3; see aligned-fp32-full-r2-20261012.
Never enable unsafe fast-math just to obtain a faster benchmark.
