#!/bin/bash
#SBATCH --job-name=llm-pers-lang-analysis
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

HOME_DIR="/home/eml/philipp.spohn"
PROJECTS_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v4"

source ${HOME_DIR}/.bashrc

cd "${PROJECTS_DIR}"

export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"

conda activate llm-pers
pip install -r requirements.txt
pip install -e .

echo "========================================"
echo "Language analysis (n-gram subset comparison)"
echo "========================================"
echo "Args: $@"

set -x
# sample_limit=null → use all 320k prompts (16k users × 20 conversations).
# Override on the command line: `sbatch sbatch/language_analysis.sbatch sample_limit=20000`.
python experiments/language_analysis/scripts/analyze.py sample_limit=null "$@"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
