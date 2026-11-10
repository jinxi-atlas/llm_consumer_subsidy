DIGITAL COUPON — STEP 1
CHFS 2019; policy context 2020.

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
chfs2019_master_202112.dta
Field: weight_hh; join key: hhid.
The fallback is read only if the weight column is absent from the primary input.

APPLICATION RULES
Select Zhejiang province and city level 2, city households before rural ones,
preserving source order within each group unless a private order is supplied.
Retain the original complete-case rule, including travel in source selection.
Travel stays out of the submitted persona prompt. Food and culture spending
are monthly; communication spending is annual. No new age filter is added.

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
  weight_hh -> weight
  city_level -> city_level
  city_lab -> city_label
  total_consump -> annual_consumption
  g1001_imp -> food_spending
  g1010_imp -> entertainment_spending
  g1009g_imp -> tele_spending
  g1018_imp -> travel_spending
  total_assets, total_liabilities, net_worth -> supplied wealth values
Demographic and weight aliases can point to existing columns; absent fields
can be obtained from the existing sources described above.

PERSONA OUTPUT COLUMNS (28)
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
  weight_hh
  city_level
  city_label
  annual_consumption
  food_spending
  entertainment_spending
  tele_spending
  travel_spending

Roster diagnostic columns are unknown when no roster was needed/read.
This is not missing persona information and does not remove a household.

STEP 2 — MAIN SURVEY SCRIPT

The complete step2_run_survey.py is independent of all other project scripts.
Input: outputs/step1/personas.csv, or --input /path/to/existing/personas.csv.
Weights and demographics are already in this file; nothing else is joined.
No sample is padded, truncated or removed by Step 2. --expected-count optionally
checks a known sample size. A missing required field stops with an explanation.

Offline prepare (the default):
  python digital_coupon/step2_run_survey.py
  python digital_coupon/step2_run_survey.py --input /path/to/personas.csv --expected-count 547
  python digital_coupon/step2_run_survey.py --self-test

Outputs for each newly prepared run, under digital_coupon/outputs/step2/run_TIMESTAMP/:
  personas.json          local frozen copy of only the needed input fields
  questionnaire.json     persona/policy/question/memory/model specification
  prompt_preview.txt     actual EDSL system/user prompts for the first persona;
                         prior answers are explicitly marked placeholders
  manifest.json          input/code hashes, sample size, signature, batch sizes
  status.json            aggregate completion counts by experiment and arm
  EXPERIMENT/answers_control.csv and EXPERIMENT/answers_treatment.csv
                         empty at preparation; filled only by explicit collection

Only when a NEW paid collection is intended:
  python digital_coupon/step2_run_survey.py --mode collect --run-dir /path/to/prepared/run
Use the existing Expected Parrot login or EXPECTED_PARROT_API_KEY in the process
environment. Never paste keys into code. DeepSeek-V3.2 via deep_infra, temperature
1, edsl==1.0.8 matches the selected historical surveys.

Collection submits each household questionnaire once per experiment/arm.
It saves batches/request.json, original results and remote job IDs locally.
Missing or invalid answers stop collection; no questions are asked again.
A started run cannot be submitted again. Analyze completed runs offline.
--batch-size overrides the default for a newly prepared run.

Offline analysis of a completed Step 2 run:
  python digital_coupon/step2_run_survey.py --mode analyze --run-dir /path/to/completed/run
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
--experiments accepts one or more of these names (default: all six):
  dining_7d, books_digital_7d, dining_6w, dining_20w,
  books_digital_6w, books_digital_20w.
Each experiment has control and treatment for the same households.

Following the submitted Appendix A, each horizon has ONE independent question:
7 days (days 1-7), 6 weeks (days 1-42), or 20 weeks (days 1-140).
There is no cross-horizon memory and no later-period question in the 7-day survey.
Answers must be finite numbers. No numerical amount minimum, maximum or
cumulative cap is applied in prompts or acceptance checks; zero is allowed.
Default batches: 100 for 7 days, 200 for longer horizons.

The persona uses head sex/age, residence/province, household size/income/annual
consumption, assets/liabilities/net worth, elderly and children, monthly food
and entertainment spending, and ANNUAL communication-equipment spending.
Travel spending is not included in the prompt. weight_hh is an analysis weight,
not a sentence in the persona. These fields come directly from Step 1.

SOURCE SELECTION
7-day dining: ate_dining_books_no_travel_persona_1019.py.
7-day books/digital: ate_dining_books_no_travel_max100k_1019.py.
Independent long horizons: ate_dining_books_horizon_robust_batchsize_1020.py.
Historical templates were embedded and then revised to match Appendix A
at the user's request. No historical script is imported at runtime.
Weighted inference follows weighted_ate_inference_selected_1020.py.

APPENDIX ALIGNMENT (2026-11-08)
The survey date restriction now also prohibits external knowledge and
real-world data. The control arrival time explicitly says approximately
the first two minutes. Age wording uses the year-based age directly.
Questions retain the Appendix's individual real-name Alipay, all-purchases,
out-of-pocket definition. Extra spending-scope clarification is removed.
Response instructions require only one number, with no unit, commas,
explanation or other text; all previous amount bounds are removed.
The 7-day questionnaire no longer asks days 8-42 or days 43-140.
This revision cannot resume an earlier questionnaire checkpoint; the
saved specification must match. Historical scripts and answers remain
unchanged. Newly collected answers under the revised specification are
not the frozen answers that generated the submitted paper's tables.

STEP 3 — SINGLE-TRANSACTION THRESHOLDS
step3_run_threshold_survey.py is self-contained. It uses the same Step 1
personas.csv and weights as Step 2, with the Appendix-aligned persona/date and
policy instructions. No new CHFS input is required.

Categories: dining, books, digital (mobile phones/accessories). The latter two
retain the complete books/mobile-phone coupon packet. Each control answers one
bill question. Each treatment answers the same pre-discount bill question,
then the coupon-use choice with memory of its bill and policy. A transaction
has already been decided and will occur; this is not a seven-day total-spending
question. At most one eligible coupon can be selected for the transaction.
Prompt templates follow Appendix A.5.1, with no appended amount limits or the
historical 'this is a choice question, not an amount question' extra sentence.
The original 1-to-100000 amount restriction is removed. Returned amounts must
be finite and nonnegative; zero is retained. There is no amount upper cap.
Choice validation preserves coupon eligibility and explicit nonuse options.

Offline preparation (default; does NOT collect answers):
  python digital_coupon/step3_run_threshold_survey.py --input /path/to/personas.csv
Optional: --expected-count 547, --categories dining books digital, --batch-size 100.
Default output: digital_coupon/outputs/step3/run_TIMESTAMP/
  personas.json, questionnaire.json, manifest.json, prompt_preview.txt
  dining/, books/, digital/ for the selected categories
The preview includes all formal questionnaire prompts for the first input
household. All these files can contain private data and stay local.

Only for an explicitly intended NEW paid survey:
  python digital_coupon/step3_run_threshold_survey.py --mode collect --run-dir /path/to/prepared/run
Collection submits each household questionnaire once per category/arm.
Missing or invalid answers stop collection. Neither the bill nor coupon choice
is asked again. A started run cannot be submitted again.
Original result batches and .job/ requests/remote IDs are saved locally.
Per-category consolidated answers are named:
  answers_control_deepseek-ai_DeepSeek-V3.2.csv
  answers_treatment_deepseek-ai_DeepSeek-V3.2.csv
They contain agent_idx, arm, model, bill_before_discount_yuan and coupon_choice. The frozen personas map agent_idx to hhid.

Offline analysis of completed results from this version:
  python digital_coupon/step3_run_threshold_survey.py --mode analyze --run-dir /path/to/completed/run
Optional: --xing-csv /path/to/existing/xing_histograms.csv adds the published
Xing histogram comparison; this is recovered aggregate figure data, not CHFS
or human microdata. Without it, the script draws the LLM panels only.
No hard-coded local source-PDF path or historical plot script is required.

Outputs under the run's analysis/analysis_TIMESTAMP/:
  transactions.csv, matched_persona_pairs.csv, weighted_histograms.csv
  threshold_bin_inference.csv, analysis_audit.json
  threshold_distributions.png/pdf (omit with --no-plots)
All households are retained in both receipt arms, including nonredeemers.
Bins are [90,100), [210,220), [50,55), [2000,2200) for the four tests;
figure bin widths are 10, 5, 200. Last bins absorb the upper tails at 240,
100 and 3000 respectively. Estimates are paired weighted differences in bin
shares, expressed in percentage points, with HC1 t(n-1) intervals and p-values.
Xing panels compare redemption-status groups, so their grouping differs from
the LLM coupon-receipt groups. The audit records this distinction.

Sources: threshold_dining_books_no_travel_max100k_1019.py for questions; plot_coupon_threshold_all_personas_1020.py for the
full-sample grouping and binning; weighted_threshold_bin_inference_1020.py for
paired inference. All required logic is embedded; no project script imports.
Historical saved answers use a different prompt signature and are preserved.
They are not silently imported as answers to this revised questionnaire.
Offline checks matched all four historical threshold-bin tests and exercised
weighted plots and zero amounts with invented data. Current checks also verify
single-pass collection and stopping on missing/invalid answers without reasking.


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
  python digital_coupon/step4_run_persona_sensitivity.py --mode collect --run-dir /path/to/step4/run
Collection submits each questionnaire once. Missing/invalid answers stop it;
there is no repair, follow-up or resubmission mechanism. Raw results stay local.

For completed answers from this Step 4 version:
  python digital_coupon/step4_run_persona_sensitivity.py --mode analyze --run-dir /path/to/step4/run
The saved variant is used automatically. The same complete household sample
must be present in both arms. Analysis writes only:
  analysis/analysis_TIMESTAMP/ate_estimates.csv
  analysis/analysis_TIMESTAMP/analysis_audit.json
The estimate file contains persona_variant, experiment, horizon, outcome,
weighting, n, estimate, standard_error, ci95_lower, ci95_upper and p_value.

Offline verification with invented records and no model requests:
  python digital_coupon/step4_run_persona_sensitivity.py --self-test
Historical scripts and answers are preserved. Their earlier prompt/answer
formats are not silently treated as completed runs of this revised script.

Digital-coupon Step 4 variants; prepare them separately with the same input:
  python digital_coupon/step4_run_persona_sensitivity.py --variant no_consumption --input /path/to/personas.csv
  python digital_coupon/step4_run_persona_sensitivity.py --variant no_persona --input /path/to/personas.csv
Default variant: no_consumption. Each run contains only the selected variant.
no_consumption removes annual total consumption, monthly food spending,
monthly cultural/entertainment spending and annual communication-device
spending. Other demographics, income, assets and liabilities stay in the prompt.
no_persona retains only the common survey context. IDs remain distinct for
sampling/answer matching and weights for analysis; neither appears in prompts.
Neither version attaches the removed information to EDSL agent traits.
Both use Step 1's hhid, agent_idx, province, area_type, gender_code, birth_year,
total_population, hh_income, total_assets, total_liabilities, net_worth,
seniors_60_or_over, children_under_14 and weight_hh. No extra file is needed.
Each category/horizon uses its own treatment and control: dining and
books/phones/accessories, at 7 days, 6 weeks and 20 weeks. Each survey asks
one cumulative spending question, with no cross-horizon memory or amount cap.
There are six weighted ATE rows per variant when all experiments finish.

Persona-removal definitions follow Table 4 and the selected historical
ate_dining_books_robustness_no_consumption_1019.py and
ate_dining_books_robustness_no_persona_separate_agents_batch100_1020.py.
This script preserves the user's Appendix-aligned date/policy/questions;
it does not restore historical amount bounds or sequential period questions.
It produces the selected variant's ATEs only, without the full-persona or
administrative-estimate columns of Table 4.
