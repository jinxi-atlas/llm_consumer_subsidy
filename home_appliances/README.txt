HOME APPLIANCES — STEP 1
CHFS 2011; policy context 2008.

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

KNOWN EXISTING WEIGHT SOURCE
农村户主样本（4）.csv
Field: swgt; join key: hhid.
The fallback is read only if the weight column is absent from the primary input.

APPLICATION RULES
Select rural households. No additional minimum-head-age filter is applied.
Keep all supplied required variables; missing core values cause an error.
Retain all 11 ownership indicators, owned/not-owned descriptions and past
durable spending for Appendix C. Later main prompts must omit those fields.
Unknown durable expenditure is retained with a flag; ownership must be 0/1.

CONFIGURATION
files.households is unset: supply --input or the actual existing filename(s).
files.members is unset: no separate member input is mandatory.
files.master references the known weight source but is only read if needed.
Column aliases describe fields in existing files, not new datasets.

PRIMARY HOUSEHOLD COLUMN ALIASES
  hhid -> hhid
  prov_CHN -> province
  area_type -> area_type
  total_population -> household_size
  hh_income -> household_income
  a2003 -> gender
  a2005 -> birth_year
  children_under_14 -> children_under_14
  seniors_60_or_over -> seniors_60_or_over
  swgt -> weight
  g1014_imp -> durable_spending
  c8001_1_mc -> owns_1
  c8001_2_mc -> owns_2
  c8001_3_mc -> owns_3
  c8001_4_mc -> owns_4
  c8001_5_mc -> owns_5
  c8001_6_mc -> owns_6
  c8001_7_mc -> owns_7
  c8001_8_mc -> owns_8
  c8001_9_mc -> owns_9
  c8001_10_mc -> owns_10
  c8001_11_mc -> owns_11
  total_assets, total_liabilities, net_worth -> supplied wealth values
Demographic and weight aliases can point to existing columns; absent fields
can be obtained from the existing sources described above.

PERSONA OUTPUT COLUMNS (36)
  hhid
  agent_idx
  province
  area_type
  region
  total_population
  hh_income
  gender_code
  birth_year
  children_under_14
  seniors_60_or_over
  roster_birth_year_missing_count
  household_size_roster
  household_size_conflict
  head_age_survey_year
  head_age_policy_year
  total_assets
  total_liabilities
  net_worth
  wealth_missing_value_count
  swgt
  last_year_durable_spending
  c8001_1_mc
  c8001_2_mc
  c8001_3_mc
  c8001_4_mc
  c8001_5_mc
  c8001_6_mc
  c8001_7_mc
  c8001_8_mc
  c8001_9_mc
  c8001_10_mc
  c8001_11_mc
  owned_durable_text
  not_owned_durable_text
  durable_spending_missing

Roster diagnostic columns are unknown when no roster was needed/read.
This is not missing persona information and does not remove a household.

STEP 2 — MAIN SURVEY SCRIPT

The complete step2_run_survey.py is independent of all other project scripts.
Input: outputs/step1/personas.csv, or --input /path/to/existing/personas.csv.
Weights and demographics are already in this file; nothing else is joined.
No sample is padded, truncated or removed by Step 2. --expected-count optionally
checks a known sample size. A missing required field stops with an explanation.

Offline prepare (the default):
  python home_appliances/step2_run_survey.py
  python home_appliances/step2_run_survey.py --input /path/to/personas.csv --expected-count 3244
  python home_appliances/step2_run_survey.py --self-test

Outputs for each newly prepared run, under home_appliances/outputs/step2/run_TIMESTAMP/:
  personas.json          local frozen copy of only the needed input fields
  questionnaire.json     persona/policy/question/memory/model specification
  prompt_preview.txt     actual EDSL system/user prompts for the first persona;
                         prior answers are explicitly marked placeholders
  manifest.json          input/code hashes, sample size, signature, batch sizes
  status.json            aggregate completion counts by experiment and arm
  EXPERIMENT/answers_control.csv and EXPERIMENT/answers_treatment.csv
                         empty at preparation; filled only by explicit collection

Only when a NEW paid collection is intended:
  python home_appliances/step2_run_survey.py --mode collect --run-dir /path/to/prepared/run
Use the existing Expected Parrot login or EXPECTED_PARROT_API_KEY in the process
environment. Never paste keys into code. DeepSeek-V3.2 via deep_infra, temperature
1, edsl==1.0.8 matches the selected historical surveys.

Collection submits each household questionnaire once per experiment/arm.
It saves batches/request.json, original results and remote job IDs locally.
Missing or invalid answers stop collection; no questions are asked again.
A started run cannot be submitted again. Analyze completed runs offline.
--batch-size overrides the default for a newly prepared run.

Offline analysis of a completed Step 2 run:
  python home_appliances/step2_run_survey.py --mode analyze --run-dir /path/to/completed/run
Both arms must contain the complete input sample. Analysis matches by hhid,
uses the frozen Step 1 weights, and writes a new analysis/analysis_TIMESTAMP/:
  ate_estimates.csv       weighted and unweighted paired effects and inference
  household_contrasts.csv household-level treatment-minus-control differences
  analysis_audit.json    input/run hashes and inference specification
  spending_ate.pdf/png   weighted spending effects and 95% intervals
Use --no-plots to omit figures. Weighted inference is household-paired HC1 with
t(n-1), as in the selected paper analysis; it is not full CHFS survey-design
inference. No final estimate is generated while questionnaires are incomplete.

All outputs contain or can reveal private persona information; outputs/ is
ignored by Git. No CHFS data, real answer records, authentication or old script
copies are distributed. There is no excluded_households.csv output.
Historical completed answers remain untouched. Do not run paid collection to
verify or reorganize code. Historical-answer/table reproduction is a later step.

EXPERIMENTS
--experiments accepts rebate, discount, or both (default: both).
The 13% reimbursement and instant-discount experiments retain separate control
draws. Each household/arm answers four questions in the original order:
six-month spending, six-month financing, two-year spending, and two-year financing.
All preceding answers remain in memory. The original fifth question (planned
appliance categories) was removed on request. The first four question texts,
persona and policy wording are unchanged.
The two-year amount is cumulative, not an additional amount after six months.
Default batch size: 100 complete questionnaires for one mechanism/arm.

The main persona uses province, head sex/age in 2008, household size/income,
assets/liabilities/net worth, elderly and children. It omits all 11 ownership
indicators and past durable spending, which Step 1 retains for Appendix C.
swgt is used for analysis, not a sentence in the persona.

Quality checks for the four retained questions preserve the original rules: six-month spending
0..10,000, two-year spending 0..20,000 and >= six-month spending, four financing
shares between 0 and 100 that sum to 100, correct zero-purchase financing,
and raw spending matching saved numeric answers. Retries now cover all four
questions; no purchase-category answer is collected or required.
The numerical questions themselves keep their original EDSL format; caps are
acceptance checks, not newly added instructions in the respondent prompt.

Analysis includes spending, purchase participation (percentage points), and
financing amounts from savings, borrowing and reduced other consumption. With
both mechanisms complete it additionally writes discount_minus_rebate.csv,
comparing the two household-paired effects. Separate subgroup tables and
Appendix C persona sensitivity will be organized in later steps.

SOURCE SELECTION
home_ate_1029_financing_discount_no_durable_persona.py supports both mechanisms;
its rebate prompts match home_ate_1027_no_durable_persona.py. Main analysis uses
the no-durable-persona specification; historical sources named baseline used
durable information and belong to the appendix instead.
Weighted inference follows appliance_persona_delivery_sensitivity_20261030.py.

FOUR-QUESTION REVISION
This version prepares four-question runs and cannot resume a five-question
checkpoint: the saved questionnaire signature must match. Historical scripts,
five-question answers and submitted-paper estimates are preserved separately.
Removing Q5 does not change the spending/financing analysis formulas or input
fields, but this revision is not a new reproduction of the historical estimates.

STEP 3 — OFFLINE HETEROGENEITY
step3_analyze_heterogeneity.py is standalone and never calls a model.
Input: the same Step 1 personas.csv (hhid, agent_idx, hh_income, swgt, region)
and completed answers for both mechanisms and both arms. It joins by hhid and
verifies agent_idx, all households and four-question answer validity.

Run with a completed Step 2 folder:
  python home_appliances/step3_analyze_heterogeneity.py --input /path/to/personas.csv --run-dir /path/to/step2/run

Or reuse four existing answer CSVs directly (original EDSL column names):
  python home_appliances/step3_analyze_heterogeneity.py --input /path/to/personas.csv --rebate-control /path/to/rebate_control.csv --rebate-treatment /path/to/rebate_treatment.csv --discount-control /path/to/discount_control.csv --discount-treatment /path/to/discount_treatment.csv
The four explicit files must be the selected matching experiment arms for the
same sample. Original fifth-question columns, if present, are not needed.
No separate registry, weight extract, intermediate contrasts or CHFS input is
required. Each raw answer file must retain agent.hhid, agent.agent_idx, the
four answer.* fields, and raw_model_response.* for the two spending amounts.

Income groups preserve the historical rule: pandas.qcut of household annual
income into five SAMPLE quintiles, with negative incomes retained. They are
not weighted-population quintiles or per-person income groups. Duplicate
quantile edges stop rather than silently changing the groups. Regions use the
four labels already supplied by Step 1. Weights normalize within each group.
Inference follows the historical paired HC1 t(n_group-1) and joint Wald
F(k-1,n-1) with global n/(n-1) covariance correction. Small/empty groups or
singular joint covariance are marked non-estimable, not assigned significance.

Default outputs: home_appliances/outputs/step3/analysis_TIMESTAMP/
  subgroup_estimates.csv, joint_heterogeneity_tests.csv
  table8_regional_effects.csv, table9_income_effects.csv
  table10_panel_d.csv, table10_panel_d_joint_test.csv
  household_groups.csv, analysis_audit.json, heterogeneity.png/pdf
The table files contain estimates, standard errors, intervals and p-values;
they are table data, not final manuscript typesetting.
--output-dir selects a NEW folder; --no-plots omits figures.
All household-level outputs remain local and ignored by Git.

Sources consolidated: appliance_reduced_persona_section3_20261030.py,
Step 2's existing answer checks, and the original sample-quintile definition
in summarize_completed_weighted_swgt.py. No source is imported at runtime.
Offline verification matched 108 historical main-specification subgroup rows
and 24 joint tests, without exporting any household records to this folder.


STEP 4 — PERSONA SENSITIVITY, ATE ONLY

step4_run_persona_sensitivity.py is self-contained. Its only new-run input is
the existing Step 1 personas.csv. It embeds the reviewed policy/questions;
it does not import Step 2 or any historical project script.
No Step 2 answer/estimate files are read. Analysis reports only this variant's
own treatment-minus-control ATEs, with CHFS weights, household-paired HC1
standard errors, t(n-1) confidence intervals and p-values. There are no baseline
columns, cross-persona differences, delivery-gap comparisons, descriptives,
heterogeneity tables or plots in Step 4.

Default prepare is offline and writes a new local outputs/step4/VARIANT/run_*/:
  personas.json, questionnaire.json, manifest.json, prompt_preview.txt, status.json
  EXPERIMENT/answers_control.csv and EXPERIMENT/answers_treatment.csv (empty)
It never generates respondent answers by default. All private outputs stay
local and ignored by Git. No new CHFS, weight or member input is required.

For an explicitly intended paid survey only:
  python home_appliances/step4_run_persona_sensitivity.py --mode collect --run-dir /path/to/step4/run
Collection submits each questionnaire once. Missing/invalid answers stop it;
there is no repair, follow-up or resubmission mechanism. Raw results stay local.

For completed answers from this Step 4 version:
  python home_appliances/step4_run_persona_sensitivity.py --mode analyze --run-dir /path/to/step4/run
The saved variant is used automatically. The same complete household sample
must be present in both arms. Analysis writes only:
  analysis/analysis_TIMESTAMP/ate_estimates.csv
  analysis/analysis_TIMESTAMP/analysis_audit.json
The estimate file contains persona_variant, experiment, horizon, outcome,
weighting, n, estimate, standard_error, ci95_lower, ci95_upper and p_value.

Offline verification with invented records and no model requests:
  python home_appliances/step4_run_persona_sensitivity.py --self-test
Historical scripts and answers are preserved. Their earlier prompt/answer
formats are not silently treated as completed runs of this revised script.

Home-appliance Step 4 variant: with_durable_information (the only variant).
  python home_appliances/step4_run_persona_sensitivity.py --input /path/to/personas.csv
Uses Step 1 household demographic/economic fields and swgt, plus its existing:
  owned_durable_text, not_owned_durable_text, last_year_durable_spending
Unknown past durable spending remains "未知"; the household is retained.
Policy, the four questions, joint six-month/two-year questionnaire, and normal
within-questionnaire memory remain as reviewed. Both rebate and discount have
their own control. Outcomes: spending, purchase participation (percentage
points), and RMB amounts funded by existing savings, borrowing/installments,
and reduced other consumption. There are 20 weighted ATE rows across the
two mechanisms, two horizons and five outcomes when all experiments finish.
The durable block follows the selected historical with-durable surveys
home_ate_v32_full_pool_quality_retry.py and home_ate_1021_delivery_prompt.py;
inference follows appliance_persona_delivery_sensitivity_20261030.py.
