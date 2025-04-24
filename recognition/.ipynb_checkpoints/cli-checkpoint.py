import sys
import argparse
from recognition.TrajectoryAnalysis import trajectory_analysis, molecular_analysis, free_energy_proton_hopping

def parse_and_execute(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run analysis functions for recognition."
    )
    subparsers = parser.add_subparsers(dest='function', required=True)

    # Subparser for trajectory_analysis
    traj_parser = subparsers.add_parser('trajectory_analysis')
    traj_parser.add_argument("traj_file_path")
    traj_parser.add_argument("--do_hydrogen_analysis", action='store_true', default=True)
    traj_parser.add_argument("--no-do_hydrogen_analysis", action='store_false', dest='do_hydrogen_analysis')
    traj_parser.add_argument("--save", action='store_true', default=True)
    traj_parser.add_argument("--no-save", action='store_false', dest='save')
    traj_parser.add_argument("--step_min", type=int, default=0)
    traj_parser.add_argument("--Nstep", type=int, default=1)
    traj_parser.add_argument("--stepsList", nargs="*", type=int, default=None)
    traj_parser.add_argument("--analyze_whole_trajectory", action='store_true', default=True)
    traj_parser.add_argument("--no-analyze_whole_trajectory", action='store_false', dest='analyze_whole_trajectory')
    traj_parser.add_argument("--do_molecular_analysis", action='store_true', default=True)
    traj_parser.add_argument("--no-do_molecular_analysis", action='store_false', dest='do_molecular_analysis')

    # Subparser for molecular_analysis
    mol_parser = subparsers.add_parser('molecular_analysis')
    mol_parser.add_argument("frames_list")
    mol_parser.add_argument("--stepsList", nargs="*", type=int, default=None)

    # Subparser for free_energy_proton_hopping
    fe_parser = subparsers.add_parser('free_energy_proton_hopping')
    fe_parser.add_argument("h_id_list", nargs="+")
    fe_parser.add_argument("mol1")
    fe_parser.add_argument("mol2")
    fe_parser.add_argument("mol3")
    fe_parser.add_argument("mol4")
    fe_parser.add_argument("--T", type=int, default=3000)
    fe_parser.add_argument("--num_bins", type=int, default=100)
    fe_parser.add_argument("--bins_range", nargs=2, type=float, default=(-1.5, 1.5))
    fe_parser.add_argument("--plot", action='store_true')
    fe_parser.add_argument("--no-plot", action='store_false', dest='plot')
    fe_parser.add_argument("--label", type=str, default="")
    fe_parser.add_argument("--save", action='store_true', default=True)
    fe_parser.add_argument("--no-save", action='store_false', dest='save')

    parsed_args = parser.parse_args(args)
    
    if parsed_args.function == 'trajectory_analysis':
        trajectory_analysis(
            parsed_args.traj_file_path,
            parsed_args.do_hydrogen_analysis,
            parsed_args.save,
            parsed_args.step_min,
            parsed_args.Nstep,
            parsed_args.stepsList,
            parsed_args.analyze_whole_trajectory,
            parsed_args.do_molecular_analysis
        )
    elif parsed_args.function == 'molecular_analysis':
        molecular_analysis(
            parsed_args.frames_list,
            parsed_args.stepsList
        )
    elif parsed_args.function == 'free_energy_proton_hopping':
        free_energy_proton_hopping(
            parsed_args.h_id_list,
            parsed_args.mol1,
            parsed_args.mol2,
            parsed_args.mol3,
            parsed_args.mol4,
            parsed_args.T,
            parsed_args.num_bins,
            tuple(parsed_args.bins_range),
            parsed_args.plot,
            parsed_args.label,
            parsed_args.save
        )

