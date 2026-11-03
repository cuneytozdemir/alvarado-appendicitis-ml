"""run_all.py - Reproduce every table and figure of the revised manuscript."""
import runpy, sys, time
STEPS = ["r01_descriptive", "r02_nested_cv", "r03_aggregate", "r04_calibration_dca",
         "r05_explain", "r06_learning_curve_samplesize"]
if __name__ == "__main__":
    for s in (sys.argv[1:] or STEPS):
        t0 = time.time(); print(f"\n### {s}")
        runpy.run_module(s, run_name="__main__")
        print(f"### {s} finished in {time.time()-t0:.0f} s")
