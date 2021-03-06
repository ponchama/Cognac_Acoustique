#!/bin/bash
set -e

# Usage:
#   bash
#   source activate cognac
#   ./launch-jlab.sh   (does not try connect to dashboard)
#   ./launch-jlab.sh  wait  (wait for node file)
#   ./launch-jlab.sh  1208493.datarmor0.nodefile  (use a specific node file)
#
#  If you want to specify an conda environment use: -e <env_name>


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--environment)
    CONDAENV="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "${#CONDAENV}" -eq 0 ]; then
    CONDAENV="cognac"
fi

if [ "${#POSITIONAL[0]}" -eq 0 ]; then
    echo Dask dashboard connection: off
    DASHINFO="0"
else
    echo Dask dashboard connection: on
    DASHINFO=$POSITIONAL[0]
fi

echo Conda environment: "${CONDAENV}"
#echo POSITIONAL = "${POSITIONAL[0]}"

source activate $CONDAENV

# create a log file with random name and delete existing jlab log file if any
JLAB_LOG="jlab.$RANDOM.log"
rm -f $JLAB_LOG > /dev/null 2>&1

echo "Launching job ..."
#s=`qsub jlab.pbs`
s=`qsub -v JLAB_LOG=$JLAB_LOG,DASHINFO=$DASHINFO,CONDAENV=$CONDAENV jlab.pbs`
# in order to have live log output, use the following qsub option: -k oe

sjob=${s%.*}
echo " ... ${s} launched"

# wait for jlab.log
while true; do
    if [ -f $JLAB_LOG ]; then
        echo "$JLAB_LOG has been found "
	cat ${JLAB_LOG}
        echo "Kill jlab job with:"
        echo "qdel ${sjob}"
        break
    fi
    sleep 1
done

