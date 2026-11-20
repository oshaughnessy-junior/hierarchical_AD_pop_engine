"""Run two_channel_pl_demo.py on a host without RIFT (cardassia).

gwkokab.errors imports RIFT.lalsimutils (for MsunInSec only) at package import, which
two_channel_pl_demo reaches through production_plpeak_sampled_recovery. This wrapper
registers a minimal stand-in, then runs the demo unchanged with the given arguments:

    python native4d_run_demo.py --mode grad --likelihood native4d ...
"""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", ".."))
WS = os.path.dirname(ENG)
# what env.sh provides on CIT: engine root (red_team), engine/src, the sampled gwkokab
sys.path[:0] = [HERE, ENG, os.path.join(ENG, "src"), os.path.join(WS, "gwkokab-sampled", "src")]
import native4d_population as NP  # noqa: E402

NP.install_rift_stub()
sys.argv = [os.path.join(HERE, "two_channel_pl_demo.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
