"""Step 3: offline regional and income heterogeneity for Tables 8, 9 and 10D.
Read Step 1 personas and completed Step 2 answers (or four existing answer CSVs).
All calculations are embedded; no project script imports and no model calls.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
SCRIPT_DIR=Path(__file__).resolve().parent

def is_missing_answer(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}

def spending_answer(value):
    if is_missing_answer(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) and number >= 0 else None

def funding_answer(value):
    """按题目顺序读取四项百分比，不填零、不归一化，也不接受带百分号的值。"""
    if is_missing_answer(value):
        return None
    try:
        shares = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return None
    if not isinstance(shares, list) or len(shares) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float, str)) for item in shares):
        return None
    try:
        shares = [float(item) for item in shares]
    except (ValueError, TypeError, OverflowError):
        return None
    if not all(np.isfinite(item) and 0 <= item <= 100 for item in shares):
        return None
    return shares if np.isclose(sum(shares), 100, rtol=0, atol=1e-6) else None

def funding_is_consistent(spending, shares):
    if spending is None or shares is None:
        return False
    return shares == [0, 0, 0, 100] if spending == 0 else shares[3] == 0

FUNDING_SPENDING = {
    "funding_share_post": "spend_point_post",
    "funding_share_two_years_post": "spend_point_two_years_post",
}

def raw_spending_number(row, question):
    """Verify an explicit amount; never infer or overwrite a saved answer."""
    import re
    value = row.get(f'raw_model_response.{question}_raw_model_response', '')
    try:
        raw = ast.literal_eval(value) if isinstance(value, str) else value
        content = raw['choices'][0]['message']['content'].strip()
    except (ValueError, SyntaxError, TypeError, KeyError, IndexError, AttributeError):
        return None, 'raw_response_unavailable'
    text = re.sub(r'^```(?:json)?\s*', '', content).strip()
    text = re.sub(r'\s*```$', '', text).strip()
    try:
        parsed = json.loads(text)
        answer = parsed.get('answer') if isinstance(parsed, dict) else parsed
        if isinstance(answer, (int, float)) and not isinstance(answer, bool) and np.isfinite(answer):
            return float(answer), 'explicit_number'
    except ValueError:
        pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    if lines and re.fullmatch(number, lines[0]):
        return float(lines[0]), 'explicit_number'
    return None, 'raw_amount_not_unambiguous'

def quality_reasons(row):
    reasons = [f'invalid_or_missing:{q}' for q in QUESTION_NAMES if question_needs_repair(row, q)]
    six = spending_answer(row.get('answer.spend_point_post', ''))
    two = spending_answer(row.get('answer.spend_point_two_years_post', ''))
    if six is not None and two is not None and two < six:
        reasons.append('two_years_below_six_months')
    if six is not None and six > QUALITY_RULES['six_month_max_rmb']:
        reasons.append('six_months_above_10000')
    if two is not None and two > QUALITY_RULES['two_year_max_rmb']:
        reasons.append('two_years_above_20000')
    for q, stored in [('spend_point_post', six), ('spend_point_two_years_post', two)]:
        if stored is None:
            continue
        explicit, status = raw_spending_number(row, q)
        if explicit is None:
            reasons.append(f'{status}:{q}')
        elif not np.isclose(explicit, stored, rtol=0, atol=1e-8):
            reasons.append(f'raw_saved_amount_mismatch:{q}')
    return reasons

def outcome_arrays(frame, experiment):
    result = {}
    for horizon, spending, funding in [
        ("six_months", "spend_point_post", "funding_share_post"),
        ("two_years", "spend_point_two_years_post", "funding_share_two_years_post"),
    ]:
        y = pd.to_numeric(frame["answer." + spending]).to_numpy(float)
        shares = np.array([funding_answer(v) for v in frame["answer." + funding]], float)
        result[horizon, "spending"] = y
        result[horizon, "purchase_participation"] = 100.0 * (y > 0)
        for j, name in enumerate(["existing_savings_amount", "borrowing_amount", "reduced_other_consumption_amount"]):
            result[horizon, name] = y * shares[:, j] / 100
        if not np.allclose(y * shares[:, :3].sum(axis=1) / 100, y):
            raise ValueError("Funding amounts do not sum to spending.")
    return result

QUESTION_NAMES=['spend_point_post','funding_share_post','spend_point_two_years_post','funding_share_two_years_post']
ANSWER_PARSERS={q:funding_answer if q.startswith('funding') else spending_answer for q in QUESTION_NAMES}
QUALITY_RULES={'six_month_max_rmb':10000,'two_year_max_rmb':20000}
def question_needs_repair(row, question_name):
    """失败指答案缺失、原验证失败或不符合既有答案格式；零支出不是失败。"""
    parser = ANSWER_PARSERS[question_name]
    value = row.get(f"answer.{question_name}", "")
    flag = str(row.get(f"validated.{question_name}_validated", "")).strip().lower()
    answer = parser(value)
    if answer is None or flag in {"false", "0", "fail", "failed"}:
        return True
    if question_name in FUNDING_SPENDING:
        spending = spending_answer(row.get(f"answer.{FUNDING_SPENDING[question_name]}", ""))
        if spending is not None:
            return not funding_is_consistent(spending, answer)
    return False

GROUPS = {'region': ['Eastern', 'Central', 'Northeastern', 'Western'],
          'income_quintile': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def infer(values, weights):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    assert len(x) > 1 and x.shape == w.shape and np.isfinite(x).all()
    assert np.isfinite(w).all() and (w > 0).all()
    n, a = len(x), w / w.sum()
    estimate = float(a @ x)
    se = float(np.sqrt(n / (n - 1) * np.sum((a * (x - estimate)) ** 2)))
    t = estimate / se if se else (0 if estimate == 0 else np.copysign(np.inf, estimate))
    margin = float(stats.t.ppf(.975, n - 1) * se)
    return dict(n_profiles=n, estimate=estimate, standard_error=se,
                ci95_lower=estimate - margin, ci95_upper=estimate + margin,
                p_value=float(2 * stats.t.sf(abs(t), n - 1)))

def joint(values, labels, weights, groups):
    x, lab, w = np.asarray(values, float), np.asarray(labels), np.asarray(weights, float)
    n, k = len(x), len(groups)
    beta = np.array([np.average(x[lab == g], weights=w[lab == g]) for g in groups])
    scores = np.column_stack([np.where(lab == g, w * (x - beta[j]), 0) / w[lab == g].sum()
                              for j, g in enumerate(groups)])
    covariance = n / (n - 1) * scores.T @ scores
    restriction = np.column_stack([-np.ones(k - 1), np.eye(k - 1)])
    delta = restriction @ beta
    variance = restriction @ covariance @ restriction.T
    f = max(0., float(delta @ np.linalg.solve(variance, delta) / (k - 1)))
    return dict(f_statistic=f, df_num=k - 1, df_den=n - 1,
                p_value=float(stats.f.sf(f, k - 1, n - 1)))

def read_personas(path):
    p = pd.read_csv(path, dtype={'hhid': str}).set_index('hhid')
    required = {'agent_idx', 'hh_income', 'swgt', 'region'}
    if required - set(p) or p.empty or p.index.has_duplicates or p.index.isna().any():
        raise ValueError('Supply unique Step 1 personas with agent_idx, hh_income, swgt and region.')
    for c in ['agent_idx', 'hh_income', 'swgt']:
        p[c] = pd.to_numeric(p[c], errors='raise')
        if not np.isfinite(p[c]).all(): raise ValueError('Nonfinite ' + c)
    if p.agent_idx.duplicated().any() or not p.agent_idx.eq(p.agent_idx.astype(int)).all():
        raise ValueError('Invalid agent_idx.')
    if (p.swgt <= 0).any() or not set(p.region).issubset(GROUPS['region']):
        raise ValueError('Invalid weights or region labels.')
    # Historical sample quintiles, NOT weighted population quintiles or per-capita income.
    p['income_quintile'], edges = pd.qcut(p.hh_income, 5, labels=GROUPS['income_quintile'], retbins=True)
    return p.sort_index(), edges


def read_answers(path, personas):
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'agent.hhid' not in f or f['agent.hhid'].duplicated().any():
        raise ValueError('Answers must have unique agent.hhid; no row-order matching.')
    f = f.set_index('agent.hhid').sort_index()
    if not f.index.equals(personas.index): raise ValueError('Answers and personas must contain the same households.')
    if not np.array_equal(pd.to_numeric(f['agent.agent_idx']), personas.agent_idx):
        raise ValueError('Answer agent_idx differs from persona identity.')
    # Also verify available frozen traits; legacy answers may use original labels.
    for column, trait in [('agent.hh_income', 'hh_income'), ('agent.swgt', 'swgt')]:
        if column in f and not np.allclose(pd.to_numeric(f[column]), personas[trait], rtol=0, atol=1e-8):
            raise ValueError('Answer traits changed: ' + trait)
    problems = [quality_reasons(row) for row in f.to_dict('records')]
    if any(problems): raise ValueError('Incomplete or invalid four-question answers; no households are dropped.')
    return outcome_arrays(f, None)


def safe_joint(values, labels, weights, groups):
    counts = [int(np.sum(labels == g)) for g in groups]
    if min(counts) < 2:
        return dict(f_statistic=np.nan, df_num=len(groups)-1, df_den=len(values)-1,
                    p_value=np.nan, status='insufficient_group_size')
    try:
        result = joint(values, labels, weights, groups)
        result['status'] = 'ok'
        return result
    except np.linalg.LinAlgError:
        return dict(f_statistic=np.nan, df_num=len(groups)-1, df_den=len(values)-1,
                    p_value=np.nan, status='singular_covariance')


def compute(personas, cells):
    rows, tests = [], []
    weights = personas.swgt.to_numpy(float)
    for horizon in ['six_months', 'two_years']:
        for outcome in ['spending', 'purchase_participation']:
            effects = {m: cells[m, 'treatment'][horizon, outcome] - cells[m, 'control'][horizon, outcome]
                       for m in ['rebate', 'discount']}
            effects['discount_minus_rebate'] = effects['discount'] - effects['rebate']
            for dimension, groups in GROUPS.items():
                labels = personas[dimension].astype(str).to_numpy()
                for mechanism, values in effects.items():
                    key = dict(dimension=dimension, horizon=horizon, outcome=outcome, mechanism=mechanism)
                    tests.append(dict(**key, **safe_joint(values, labels, weights, groups)))
                    for group in groups:
                        mask = labels == group
                        if mask.sum() < 2:
                            estimate = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
                            result = dict(n_profiles=int(mask.sum()), estimate=estimate, standard_error=np.nan,
                                          ci95_lower=np.nan, ci95_upper=np.nan, p_value=np.nan, status='insufficient_group_size')
                        else:
                            result = dict(**infer(values[mask], weights[mask]), status='ok')
                        rows.append(dict(**key, group=group, **result))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def write_results(personas, cells, edges, output, hashes, plots=True):
    groups, tests = compute(personas, cells)
    output.mkdir(parents=True, exist_ok=False)
    groups.to_csv(output / 'subgroup_estimates.csv', index=False)
    tests.to_csv(output / 'joint_heterogeneity_tests.csv', index=False)
    groups.query("mechanism == 'rebate' and dimension == 'region'").to_csv(output / 'table8_regional_effects.csv', index=False)
    groups.query("mechanism == 'rebate' and dimension == 'income_quintile'").to_csv(output / 'table9_income_effects.csv', index=False)
    panel = groups.query("dimension == 'income_quintile' and horizon == 'six_months' and outcome == 'spending'")
    panel.to_csv(output / 'table10_panel_d.csv', index=False)
    tests.query("dimension == 'income_quintile' and horizon == 'six_months' and outcome == 'spending' and mechanism == 'discount_minus_rebate'").to_csv(output / 'table10_panel_d_joint_test.csv', index=False)
    personas[['agent_idx', 'region', 'hh_income', 'swgt', 'income_quintile']].to_csv(output / 'household_groups.csv')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, dimension in zip(axes, GROUPS):
            for shift, horizon in [(-.1, 'six_months'), (.1, 'two_years')]:
                f = groups.query("dimension == @dimension and mechanism == 'rebate' and outcome == 'spending' and horizon == @horizon").set_index('group').reindex(GROUPS[dimension])
                ax.errorbar(np.arange(len(f))+shift, f.estimate.to_numpy(),
                            yerr=np.vstack([f.estimate-f.ci95_lower, f.ci95_upper-f.estimate]), fmt='o', label=horizon)
            ax.set_xticks(np.arange(len(GROUPS[dimension])), GROUPS[dimension], rotation=20)
            ax.axhline(0, color='grey', lw=.6); ax.set_ylabel('Spending effect (RMB), 95% CI'); ax.legend()
        fig.tight_layout(); fig.savefig(output/'heterogeneity.png', dpi=180); fig.savefig(output/'heterogeneity.pdf'); plt.close(fig)
    (output/'analysis_audit.json').write_text(json.dumps(dict(input_sha256=hashes, n_households=len(personas),
        income_quintile_edges=edges.tolist(), income_grouping='unweighted sample qcut, household annual income',
        inference='within-group normalized swgt; paired HC1 t(n_g-1); joint Wald F(k-1,n-1), global HC1 correction',
        non_estimable_tests=int(tests.status.ne('ok').sum()), model_calls=0), indent=2))
    return groups, tests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SCRIPT_DIR/'outputs/step1/personas.csv')
    parser.add_argument('--run-dir', type=Path, help='Completed Step 2 run containing both mechanisms.')
    for mechanism in ['rebate', 'discount']:
        for arm in ['control', 'treatment']:
            parser.add_argument('--'+mechanism+'-'+arm, type=Path, help='Existing saved answers; supply all four instead of --run-dir.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    explicit = {(m,a): getattr(args,m+'_'+a) for m in ['rebate','discount'] for a in ['control','treatment']}
    if args.run_dir and any(explicit.values()): parser.error('Choose --run-dir or four existing answer files.')
    if not args.run_dir and not all(explicit.values()): parser.error('Supply --run-dir or all four existing answer files.')
    p, edges = read_personas(args.input)
    paths = dict(explicit)
    hashes = {str(args.input): sha(args.input)}
    if args.run_dir:
        frozen_path = args.run_dir/'personas.json'
        frozen = pd.DataFrame(json.loads(frozen_path.read_text())).set_index('hhid').sort_index()
        if not frozen.index.equals(p.index): raise ValueError('Step 1 differs from frozen Step 2 households.')
        for col in ['agent_idx','hh_income','swgt']:
            if not np.array_equal(p[col], frozen[col]): raise ValueError('Frozen Step 2 mismatch: '+col)
        hashes[str(frozen_path)] = sha(frozen_path)
        paths = {(m,a):args.run_dir/m/('answers_'+a+'.csv') for m,a in paths}
    cells = {}
    for key,path in paths.items():
        hashes[str(path)] = sha(path); cells[key] = read_answers(path,p)
    destination = args.output_dir or SCRIPT_DIR/'outputs/step3'/datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f')
    write_results(p,cells,edges,destination,hashes,not args.no_plots)
    if any(sha(Path(path)) != h for path,h in hashes.items()): raise ValueError('Input changed during analysis.')
    print('Offline Step 3 results:', destination)


if __name__ == '__main__':
    main()
