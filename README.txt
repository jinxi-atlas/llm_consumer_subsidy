LLM CONSUMER SUBSIDY — STEPS 1 TO 4

home_appliances/step1_extract_personas.py — CHFS 2011; policy context 2008.
digital_coupon/step1_extract_personas.py — CHFS 2019; policy context 2020.

USE EXISTING INPUTS
Pass the existing household data file(s) through --input. Do not create new
household, member or weight extracts to satisfy this program's organization.
The primary filename is deliberately unset in the template rather than invented.
Map the existing column names in the JSON and set input_mapping_confirmed=true.

Fields already in the primary input take priority. An existing member file is
read only when head gender, birth year or required age-composition columns are
absent. An existing master/source table is read only when weights or explicitly
requested household columns are absent. Redundant configured sources are not
opened. Missing values in an existing column are not silently overwritten from
another source; the documented complete-case/error rules apply.

The same physical file can serve several roles without being read twice.
Multiple primary household inputs can be passed together; for example, the
original urban and rural data files. If residence is implicit in the source,
household_source_constants can supply its label for a named file, but will
never overwrite a residence column already present.

WEALTH AND AGE
Read supplied total_assets, total_liabilities and net_worth (or configured
aliases). Wealth construction stays outside this code. The supplied values
are kept after explicit unit conversion and an accounting-identity check.
Elderly means age>=60, output seniors_60_or_over; children means age<14.
Existing counts must already use those definitions. An old strict age>60 count
cannot simply be renamed: use the existing roster if the inclusive count is
absent. The code never guesses how many members are exactly age 60.

OUTPUT FILES
  personas.csv — household personas for subsequent steps.
  step1_audit.json — source hashes, settings, sample counts and diagnostics.
No individual household-removal CSV is generated. Existing output directories
are never overwritten. Outputs stay local and are ignored by Git.

OPTIONAL EXISTING SOURCES
--member-source: an existing roster; used only for absent demographic columns.
--master-source: an existing master/source table; used only for absent fields.
--sample-ids: an existing private list, if an explicit sample order is required.
--expected-count: verify the count without padding or truncating the sample.
There is no requirement to produce any of these files if current inputs suffice.

RUN
Review the input column aliases, then run the application script with --input
pointing to your existing file(s). --input-dir resolves relative source names;
--output-dir overrides the output location. Explicit --input paths are resolved
from the working directory. Output defaults to outputs/step1 beside the script.
Use --self-test to check invented records only, without CHFS files.

HISTORICAL WEIGHT SOURCES
Appliances: 农村户主样本（4）.csv, field swgt.
Coupons: chfs2019_master_202112.dta, field weight_hh.
These are existing source references, not bundled data. When the weight is
already in the primary input, the fallback is not needed or read. No separate
weight file is required, created or introduced by this package.

SAMPLE RULES
Keep the documented coupon geographic and complete-case selection. Appliances
use the rural input and stop on missing required core fields. The previously
introduced blanket adult-head filter has been removed; there is no added
age>=18 sample restriction.

INSTALL AND TEST
  python -m pip install -r requirements.txt
  python home_appliances/step1_extract_personas.py --self-test
  python digital_coupon/step1_extract_personas.py --self-test
Tests cover a single existing input with all fields present and nonexistent
auxiliary sources, necessary existing-source joins, and the inclusive age-60
boundary. No actual CHFS data, personal paths or historical scripts are bundled.
See INPUT_OUTPUT.txt and each application's README.txt for the field contract.

STEP 2 — MAIN SURVEYS

home_appliances/step2_run_survey.py
  Main persona, without durable ownership/past durable spending.
  Separate reimbursement and instant-discount experiments, each with its own
  no-policy control. Four questions with full memory: six-month spending,
  financing shares, two-year spending, and financing shares. The final purchase-
  category question was removed on request; historical answers remain unchanged.

digital_coupon/step2_run_survey.py
  Dining and books/phones/accessories, at 7 days, 6 weeks and 20 weeks.
  Aligned to the submitted Appendix A: one independent cumulative question
  per horizon, without cross-horizon memory or numeric amount bounds. Date
  restrictions, control timing and response wording follow the appendix.

Both scripts read only their Step 1 personas.csv. Existing weights are used
directly (swgt or weight_hh). No additional CHFS/weight/member input is required.
Each script embeds its own templates, questions, policies, single-pass collection,
weighted analysis and plotting. Neither imports another
project script. Threshold and persona-sensitivity experiments are later steps.

DEFAULT IS OFFLINE
  python home_appliances/step2_run_survey.py
  python digital_coupon/step2_run_survey.py

The commands above prepare questionnaires, a local first-persona prompt preview,
empty answer files and a checkpoint under the application's outputs/step2/.
No respondent answers are generated. --mode collect is the explicit paid mode;
--mode analyze reads completed answers from a specified Step 2 --run-dir offline.
See each application's README for exact commands and outputs.

Python 3.12 and edsl==1.0.8 were used for offline verification. Install requirements
before running Step 2. --self-test uses invented records only, with no API calls.
Retain existing completed surveys; code organization does not require rerunning
them. Historical-answer import/table replication will be organized in later
steps. This Step 2 checkpoint format is for this standalone script's own runs.
Newly collected answers do not replace the submitted paper's frozen answers.
User-requested revisions include inclusive age>=60 personas, removal of appliance
Q5, and alignment of digital-coupon prompts and question structure to Appendix A.
Historical scripts and completed answers are preserved separately. Earlier
questionnaire checkpoints cannot be resumed with a changed specification.

STEP 3 — APPLICATION-SPECIFIC ANALYSES
home_appliances/step3_analyze_heterogeneity.py:
  Offline regional/income effects and joint tests for Tables 8, 9 and 10D.
  Uses Step 1 personas and four completed Step 2 answer files.
digital_coupon/step3_run_threshold_survey.py:
  Three single-transaction categories; control bill, treatment bill + coupon use.
  Default offline prepare; explicit collect; offline weighted histograms/tests.
  Optional existing published histogram CSV adds the Figure 1 human comparison.
Both files run independently; neither loads another project script. See each
application README for commands, input fields, outputs and source provenance.
All survey scripts use a single collection pass. Missing or invalid answers
stop collection; there are no automatic follow-up questions or resubmissions.
No CHFS data or historical household responses are bundled.

STEP 4 — PERSONA SENSITIVITY, WITHIN-SPECIFICATION ATEs ONLY
home_appliances/step4_run_persona_sensitivity.py:
  Adds durable ownership/nonownership and past durable spending to personas.
  Reports ATEs for spending, participation and financing in both mechanisms.
digital_coupon/step4_run_persona_sensitivity.py:
  --variant no_consumption or --variant no_persona (separate runs).
  Reports the selected variant's spending ATEs at 7 days, 6 weeks and 20 weeks.
Both read Step 1 personas only for preparation, and their own saved answers for
analysis. Neither reads, reports or compares Step 2 results. Both are standalone,
default to offline prepare, and have no failed-answer resubmission mechanism.
