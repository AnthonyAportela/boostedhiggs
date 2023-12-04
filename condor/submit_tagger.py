#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantilla, Raghav Kansal, Farouk Mokhtar
"""
import argparse
import os


def main(args):
    try:
        proxy = os.environ["X509_USER_PROXY"]
    except ValueError:
        print("No valid proxy. Exiting.")
        exit(1)

    locdir = "condor/" + args.tag + "_" + args.year
    username = os.environ["USER"]
    homedir = f"/store/user/{username}/boostedhiggs/"
    outdir = homedir + args.tag + "_" + args.year + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    # submit a cluster of jobs per sample
    sample = args.sample
    print(f"Making directory /eos/uscms/{outdir}/{sample}")
    os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

    # make executable file
    sh_templ_file = open("condor/submit.jdl")
    eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"
    eosoutput_pkl = f"{eosoutput_dir}/"

    localsh = f"{locdir}/{sample}.sh"
    try:
        os.remove(localcondor)
        os.remove(localsh)
        os.remove(f"{locdir}/*.log")
    except Exception:
        pass
    sh_file = open(localsh, "w")
    for line in sh_templ_file:
        line = line.replace("SCRIPTNAME", args.script)
        line = line.replace("YEAR", args.year)
        line = line.replace("NUMJOBS", args.n)
        line = line.replace("STARTI", args.starti)
        line = line.replace("SAMPLE", sample)
        line = line.replace("EOSOUTPKL", eosoutput_pkl)

        sh_file.write(line)
    sh_file.close()
    sh_templ_file.close()

    os.system(f"chmod u+x {localsh}")
    if os.path.exists("%s.log" % localcondor):
        os.system("rm %s.log" % localcondor)

    # submit
    if args.submit:
        print("Submit ", localcondor)
        os.system("condor_submit %s" % localcondor)


if __name__ == "__main__":
    """
    python condor/submit.py --year 2017 --tag test
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", dest="script", default="run.py", help="script to run", type=str)
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument("--test", dest="test", action="store_true", help="only 2 jobs per sample will be created")
    parser.add_argument("--submit", dest="submit", action="store_true", help="submit jobs when created")
    parser.add_argument("--starti", dest="starti", default=0, help="start index of files", type=str)
    parser.add_argument("--n", dest="n", default=-1, help="number of files to process", type=str)
    parser.add_argument("--sample", dest="sample", default=None, help="specify sample", type=str)

    parser.set_defaults(inference=True)
    args = parser.parse_args()

    main(args)