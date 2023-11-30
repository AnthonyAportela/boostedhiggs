#!/bin/bash

### https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/combine/run_blinded.sh

####################################################################################################
# Script for fits
#
# 1) Combines cards and makes a workspace (--workspace / -w)
# 2) Background-only fit (--bfit / -b)
# 3) Expected asymptotic limits (--limits / -l)
# 4) Expected significance (--significance / -s)
# 5) Fit diagnostics (--dfit / -d)
# 6) GoF on data (--gofdata / -g)
# 7) GoF on toys (--goftoys / -t),
# 8) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impactsf $nuisance), collect (--impactsc $nuisances)
# 9) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
#
# Specify seed with --seed (default 42) and number of toys with --numtoys (default 100)
#
# Usage ./run_blinded.sh [-wblsdgt] [--numtoys 100] [--seed 42]
####################################################################################################


####################################################################################################
# Read options
####################################################################################################

workspace=0
bfit=0
limits=0
significance=0
dfit=0
dfit_asimov=0
gofdata=0
goftoys=0
impactsi=0
impactsf=0
impactsc=0
seed=444
numtoys=100
bias=-1
mintol=0.5 # --cminDefaultMinimizerTolerance
# maxcalls=1000000000  # --X-rtd MINIMIZER_MaxCalls

options=$(getopt -o "wblsdrgti" --long "workspace,bfit,limits,significance,dfit,dfitasimov,resonant,gofdata,goftoys,impactsi,impactsf:,impactsc:,bias:,seed:,numtoys:,mintol:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        -s|--significance)
            significance=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        --dfitasimov)
            dfit_asimov=1
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        -i|--impactsi)
            impactsi=1
            ;;
        --impactsf)
            shift
            impactsf=$1
            ;;
        --impactsc)
            shift
            impactsc=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --mintol)
            shift
            mintol=$1
            ;;
        --bias)
            shift
            bias=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

echo "Arguments: workspace=$workspace bfit=$bfit limits=$limits \
significance=$significance dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys"



####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is turned off)
####################################################################################################

dataset=data_obs
cards_dir="/uscms/home/fmokhtar/nobackup/boostedhiggs/combine/templates/v30/datacards"
cp ${cards_dir}/testModel.root testModel.root # TODO: avoid this
CMS_PARAMS_LABEL="CMS_HWW_boosted"

# ####################################################################################################
# # Combine cards, text2workspace, fit, limits, significances, fitdiagnositcs, GoFs
# ####################################################################################################
# # # need to run this for large # of nuisances
# # # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735


# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

logsdir=${outdir}/logs
mkdir -p $logsdir
chmod +x ${logsdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

sr2="SR1ggFpt250to300"
sr3="SR1ggFpt300to450"
sr4="SR1ggFpt450toInf"
sr5="SR2ggFpt250toInf"

sr2B="SR1ggFpt250to300Blinded"
sr3B="SR1ggFpt300to450Blinded"
sr4B="SR1ggFpt450toInfBlinded"
sr5B="SR2ggFpt250toInfBlinded"

cr1="WJetsCR"
cr1B="WJetsCRBlinded"

ccargs="SR2=${cards_dir}/${sr2}.txt SR3=${cards_dir}/${sr3}.txt SR4=${cards_dir}/${sr4}.txt SR5=${cards_dir}/${sr5}.txt"
ccargs+=" SR2B=${cards_dir}/${sr2B}.txt SR3B=${cards_dir}/${sr3B}.txt SR4B=${cards_dir}/${sr4B}.txt SR5B=${cards_dir}/${sr5B}.txt"
ccargs+=" CR=${cards_dir}/${cr1}.txt"
ccargs+=" CRB=${cards_dir}/${cr1B}.txt"

maskunblindedargs="mask_SR2=1,mask_SR3=1,mask_SR4=1,mask_SR5=1,mask_CR=1,mask_SR2B=0,mask_SR3B=0,mask_SR4B=0,mask_SR5B=0,mask_CRB=0"
maskblindedargs="mask_SR2=0,mask_SR3=0,mask_SR4=0,mask_SR5=0,mask_CR=0,mask_SR2B=1,mask_SR3B=1,mask_SR4B=1,mask_SR5B=1,mask_CRB=1"

# freeze qcd params in blinded bins
setparamsblinded=""
freezeparamsblinded=""
for bin in {2..5}
do
    setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin}=0,"
    freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin},"
done
# remove last comma
setparamsblinded=${setparamsblinded%,}
freezeparamsblinded="${freezeparamsblinded}"


if [ $workspace = 1 ]; then
    echo "Combining cards:"
    for file in $ccargs; do
    echo "  ${file##*/}"
    done
    echo "-------------------------"
    combineCards.py $ccargs > $combined_datacard

    echo "Running text2workspace"
    text2workspace.py $combined_datacard --channel-masks -o $ws 2>&1 | tee $logsdir/text2workspace.txt
else
    if [ ! -f "$ws" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi


if [ $bfit = 1 ]; then

    echo "Blinded background-only fit"
    combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${ws} \
    --setParameters ${maskunblindedargs},${setparamsblinded},r=0  \
    --freezeParameters r,${freezeparamsblinded} 2>&1 | tee $logsdir/MultiDimFit.txt

fi


if [ $significance = 1 ]; then
    echo "Expected significance"

    wsm_snapshot=higgsCombineTest.MultiDimFit.mH125
    combine -M Significance -d ${wsm_snapshot}.root -m 125 --snapshotName MultiDimFit \
    -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
    --setParameters $maskblindedargs,r=1 --floatParameters ${freezeparamsblinded},r --toysFrequentist 2>&1 | tee $logsdir/Significance.txt

fi

if [ $dfit_asimov = 1 ]; then

    wsm_snapshot=higgsCombineTest.MultiDimFit.mH125
    echo "Fit Diagnostics"
    combine -M FitDiagnostics -m 125 -d ${wsm_snapshot}.root --snapshotName MultiDimFit \
    -t -1 --expectSignal=1 --toysFrequentist --bypassFrequentistFit --saveWorkspace --saveToys \
    --setParameters $maskblindedargs --floatParameters ${freezeparamsblinded},r \
    -n Asimov --ignoreCovWarning \
    --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

fi
