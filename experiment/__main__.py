import argparse
import multiprocessing

from experiment.exp_with_pp import run_with_postprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, help='number of processes to use', default=multiprocessing.cpu_count())
    parser.add_argument('--output-dir', required=True, help='Directory for output data')
    parser.add_argument('--no-postprocessing', help='deactivates postprocessing (equivalent to original StatDP, requires --alg)', action="store_true")
    parser.add_argument('--alg', type=str, help='algorithm to evaluate (all if not specified)')
    parser.add_argument('--postfix', type=str, help='postfix for logs', default="")
    args = parser.parse_args()
    n_processes = args.processes

    if args.no_postprocessing:
        raise NotImplementedError
    else:
        run_with_postprocessing(n_processes, only_mechanism=args.alg, postfix=args.postfix, out_dir=args.output_dir)
