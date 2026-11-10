"""Step 4: persona sensitivity, within-specification ATEs only.

Adds the existing Step 1 durable ownership/nonownership and past durable spending.
Four questions cover six-month and two-year spending/financing, for reimbursement
and instant discount, each with its own control.
Input: Step 1 personas.csv, including the existing household sampling weight.
Default prepare is offline; only --mode collect submits paid questionnaires.
Single-pass collection stops on missing/invalid answers; it never reasks them.
Analysis reads this Step 4 run's own control/treatment answers. It does not
read or output Step 2 estimates, cross-specification contrasts or heterogeneity.
All required persona/policy/question, collection and analysis code is embedded.
No project script is imported or dynamically loaded. Historical files stay intact.
"""
import argparse
import ast
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
MODEL_SERVICE = "deep_infra"
MODEL_TEMPERATURE = 1
MODEL_PARAMETERS = {"temperature": 1, "max_tokens": 1000, "top_p": 1,
                    "frequency_penalty": 0, "presence_penalty": 0,
                    "logprobs": False, "top_logprobs": 3, "reasoning_effort": None}
REQUIRED_EDSL_VERSION = "1.0.8"
ARMS = ("control", "treatment")
SAVE_ENCODING = "utf-8-sig"


def load_edsl():
    """Import the pinned questionnaire renderer; no model is instantiated here."""
    global Agent, AgentList, QuestionNumerical, Scenario, ScenarioList, Survey
    if version("edsl") != REQUIRED_EDSL_VERSION:
        raise RuntimeError("Use edsl==1.0.8 to preserve the questionnaire rendering.")
    from edsl import Agent, AgentList, QuestionNumerical, Scenario, ScenarioList, Survey


APPLICATION = "home_appliances"
WEIGHT_COLUMN = "swgt"
POLICY_YEAR = 2008


VARIANTS = ('with_durable_information',)
PERSONA_VARIANT = VARIANTS[0]


def set_persona_variant(variant):
    global PERSONA_VARIANT
    if variant not in VARIANTS:
        raise ValueError(f'Unknown Step 4 persona variant: {variant}')
    PERSONA_VARIANT = variant


def clean_count(value):
    try:
        value = float(value)
        if not np.isfinite(value):
            return 0
        return max(int(round(value)), 0)
    except (TypeError, ValueError):
        return 0


def describe_seniors(count):
    if count == 0:
        return "没有60岁及以上的老人"
    return f"有{count}位60岁及以上的老人"


def describe_children(count):
    if count == 0:
        return "没有14岁以下的儿童"
    return f"有{count}名14岁以下的儿童"


def describe_gender(value):
    try:
        value = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return "性别信息未知"
    if value == 1:
        return "男性"
    if value == 2:
        return "女性"
    return "性别信息未知"


def describe_age_in_2008(value):
    try:
        birth_year = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return "年龄信息未知"
    # 沿用 threshold_1009.py 的有效出生年规则，参考年改为 2008。
    if birth_year <= 1900 or birth_year > POLICY_YEAR:
        return "年龄信息未知"
    return f"{POLICY_YEAR}年时{POLICY_YEAR - birth_year}岁"


def render_persona(agent):
    seniors_text = describe_seniors(clean_count(agent.traits.get("seniors_60_or_over", 0)))
    children_text = describe_children(clean_count(agent.traits.get("children_under_14", 0)))
    gender_text = describe_gender(agent.traits.get("gender_code", 0))
    age_text = describe_age_in_2008(agent.traits.get("birth_year", 0))
    return f"""
你是位于中国 {agent.traits.get("province", "未知省份")} 的一个农村户主。
你的性别为{gender_text}，{age_text}。

你的家庭共有{agent.traits.get("total_population", "未知")}人。
你的家庭年收入为{agent.traits.get("hh_income", "未知")}元。
你的家庭总资产为{agent.traits.get("total_assets", "未知")}元，
家庭总负债为{agent.traits.get("total_liabilities", "未知")}元，
家庭净资产为{agent.traits.get("net_worth", "未知")}元。

你的家庭中{seniors_text}，并且{children_text}。

在11类耐用品中：
你的家庭目前拥有：{agent.traits.get("owned_durable_text", "未知")}。
你的家庭目前未拥有：{agent.traits.get("not_owned_durable_text", "未知")}。

去年，你家购买家庭耐用品的支出有{agent.traits.get("last_year_durable_spending", "未知")}元。

当前是 2008 年。请基于当时的物价水平、消费水平以及上述个人情况来回答接下来的问题。
不要使用2008年之后发生的信息，以及任何外部知识或现实世界的数据。
"""


def add_traits_presentation(agent_list):
    from edsl.agents import Agent, AgentList

    return AgentList([
        Agent(traits=agent.traits, traits_presentation_template=render_persona(agent))
        for agent in agent_list
    ])


CONTROL_POLICY = "目前是2008年。在您所在的地区，未施行家电下乡政策。"


TREATMENT_POLICY_COMMON = """
目前是2008年 ，您所在的地区正在实施 “家电下乡”政策。
- 补贴范围：九大类家电产品：彩电、冰箱/冷柜、洗衣机、手机、电脑、空调、热水器（储水式/燃气/太阳能）、微波炉、电磁炉。
- 补贴标准：购买以上产品时，可获得销售价格 13%的财政补贴。
- 购买数量限制：每个农户在每个品类最多可享受补贴 2 台。
- 价格限制（超过部分不予补贴）：彩电 ≤ 3500 元, 冰箱/冷柜 ≤ 2500 元, 洗衣机 ≤ 2000 元, 手机 ≤ 1000 元, 电脑 ≤ 3500 元, 空调 ≤ 3500 元, 热水器 ≤ 2500 元, 微波炉 ≤ 1000 元, 电磁炉 ≤ 300 元。
"""


PAYMENT_TEXT = {'rebate': '- 补贴方式：采取“先购买、后补贴”的返还机制，即农户先全额付款，再申请财政返还补贴。具体而言，具有农业户口的所有人员在指定网点购买家电下乡产品后，可凭身份证明和购买发票，到乡镇财政部门申报补贴。\n  财政部门核实后，30个工作日内将补贴资金拨付至存折账户。\n', 'discount': '- 补贴方式：采取“购买时立即减价”的方式。具体而言，具有农业户口的所有人员在指定网点购买家电下乡产品时，补贴金额按上述补贴标准和价格限制计算，并在结账时直接从应付金额中扣除。\n'}


def build_scenarios(mechanism="rebate"):
    from edsl.scenarios import Scenario, ScenarioList

    if mechanism not in PAYMENT_TEXT:
        raise ValueError(f"未知补贴方式：{mechanism}")
    return {
        "control": ScenarioList([Scenario({
            "treatment_name": "Control_NoPolicy",
            "policy_info": CONTROL_POLICY,
        })]),
        "treatment_13pct": ScenarioList([Scenario({
            "treatment_name": "Treatment_13_Percent",
            "policy_info": TREATMENT_POLICY_COMMON + PAYMENT_TEXT[mechanism],
        })]),
    }


QUESTION_TEXTS = {
    'spend_point_post': """
        {{ policy_info }}
        家电范围包括以下九大类家电产品：彩电、冰箱/冷柜、洗衣机、手机、电脑、空调、热水器（储水式/燃气/太阳能）、微波炉、电磁炉。
        您预计从现在起六个月内，您的家庭会花费多少元人民币用于购买上述类别的新的家电？
        """,
    'funding_share_post': """
            根据您在第一题中对从现在起六个月内购买家电的计划，
            请估计这笔家电支出中，来自以下来源的**百分比**。
            *** 请注意 ***
            - 若您计划的家电支出为 0 元，请将 **100%** 填在“无购买计划”这一项，其余项均为 0%。
            - 否则，请根据实际情况在前三项中分配比例，且“无购买计划”必须为 0%。
            1) 来自家庭现有储蓄：_____%
            2) 通过借款/分期（含信用卡、亲友借款、小额贷款等）：_____%
            3) 通过减少其他消费（如减少外出就餐、衣物、耐用品、娱乐等）：_____%
            4) 无购买计划：_____%

            ***填写要求***
            - 每项均为0~100之间的小数或整数；
            - 四项之和 = 100；
            - 只填数字，不要写“%”。
            """,
    'spend_point_two_years_post': """
        您预计从现在起两年内，您的家庭会花费多少元人民币用于购买第一题中所列类别的新的家电？
        """,
    'funding_share_two_years_post': """
            根据您在第三题中对从现在起两年内购买家电的计划，
            请估计这笔家电支出中，来自以下来源的**百分比**。
            *** 请注意 ***
            - 若您计划的家电支出为 0 元，请将 **100%** 填在“无购买计划”这一项，其余项均为 0%。
            - 否则，请根据实际情况在前三项中分配比例，且“无购买计划”必须为 0%。
            1) 来自家庭现有储蓄：_____%
            2) 通过借款/分期（含信用卡、亲友借款、小额贷款等）：_____%
            3) 通过减少其他消费（如减少外出就餐、衣物、耐用品、娱乐等）：_____%
            4) 无购买计划：_____%

            ***填写要求***
            - 每项均为0~100之间的小数或整数；
            - 四项之和 = 100；
            - 只填数字，不要写“%”。
            """,
}


def build_survey():
    from edsl.questions import QuestionList, QuestionNumerical
    from edsl.surveys import Survey

    questions = []
    for name, prompt in QUESTION_TEXTS.items():
        if name in {"spend_point_post", "spend_point_two_years_post"}:
            question = QuestionNumerical(question_name=name, question_text=prompt)
        else:
            question = QuestionList(question_name=name, question_text=prompt)
        questions.append(question)
    return Survey(questions=questions).set_full_memory_mode()


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


ANSWER_PARSERS = {
    "spend_point_post": spending_answer,
    "spend_point_two_years_post": spending_answer,
    "funding_share_post": funding_answer,
    "funding_share_two_years_post": funding_answer,
}


FUNDING_SPENDING = {
    "funding_share_post": "spend_point_post",
    "funding_share_two_years_post": "spend_point_two_years_post",
}


def question_is_invalid(row, question_name):
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


QUALITY_RULES = {
    "two_years_must_be_at_least_six_months": True,
    "six_month_max_rmb": 10000,
    "two_year_max_rmb": 20000,
    "require_raw_spending_to_match_saved_answer": True,
    "schema_version": 1,
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
    reasons = [f'invalid_or_missing:{q}' for q in QUESTION_NAMES if question_is_invalid(row, q)]
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

EXPERIMENTS = {"rebate": "reimbursement", "discount": "instant_discount"}
QUESTION_NAMES = list(QUESTION_TEXTS)
PERSONA_FIELDS = [
    "hhid", "agent_idx", "province", "gender_code", "birth_year", "total_population",
    "hh_income", "total_assets", "total_liabilities", "net_worth",
    "seniors_60_or_over", "children_under_14", "swgt",
    "owned_durable_text", "not_owned_durable_text", "last_year_durable_spending",
]
TEXT_FIELDS = {"hhid", "province", "owned_durable_text", "not_owned_durable_text", "last_year_durable_spending"}


def make_agents(data):
    return add_traits_presentation(AgentList([Agent(traits=row) for row in data.to_dict("records")]))


def make_scenario(experiment, arm):
    return build_scenarios(experiment)["control" if arm == "control" else "treatment_13pct"][0]


def make_survey(experiment, arm):
    return build_survey()


def question_names(experiment):
    return QUESTION_NAMES


def parse_answer(question, value):
    return ANSWER_PARSERS[question](value)


def response_problems(row, experiment):
    return quality_reasons(row)


def default_batch_size(experiment):
    return 100


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


def example_answers(experiment, treatment=False, offset=0):
    six = 100 + offset + (20 if treatment else 0)
    return {"spend_point_post": six, "funding_share_post": [60, 10, 30, 0],
            "spend_point_two_years_post": six + 200,
            "funding_share_two_years_post": [50, 30, 20, 0]}


def application_self_test(data):
    agent = make_agents(data)[0]
    assert "60岁及以上" in agent.traits_presentation_template
    for phrase in ["目前拥有", "目前未拥有", "去年，你家购买家庭耐用品"]:
        assert phrase in agent.traits_presentation_template
    row = synthetic_response(agent, "rebate", "control", example_answers("rebate"))
    assert not quality_reasons(row)
    row["answer.funding_share_post"] = "[70, 20, 30, 0]"
    assert quality_reasons(row)
    row = synthetic_response(agent, "rebate", "control", example_answers("rebate"))
    row["answer.spend_point_two_years_post"] = "1"
    assert "two_years_below_six_months" in quality_reasons(row)
    row = synthetic_response(agent, "rebate", "control", example_answers("rebate"))
    row["raw_model_response.spend_point_post_raw_model_response"] = repr({"choices": [{"message": {"content": "999"}}]})
    assert "raw_saved_amount_mismatch:spend_point_post" in quality_reasons(row)
    answers = example_answers("rebate")
    answers.update(spend_point_post=0, funding_share_post=[0, 0, 0, 100],
                   spend_point_two_years_post=0, funding_share_two_years_post=[0, 0, 0, 100])
    assert not quality_reasons(synthetic_response(agent, "rebate", "control", answers))


def run_remote_batch(job, batch_dir):
    """Submit this batch once and save its original result."""
    from edsl import Coop
    from edsl.jobs.remote_inference import JobsRemoteInferenceHandler
    raw_path, state_path = batch_dir / 'remote_results.csv', batch_dir / 'remote_job.json'
    if raw_path.exists() or state_path.exists():
        raise RuntimeError(f'This batch has already been submitted; collection stopped: {batch_dir}')
    coop = Coop()
    if not coop.api_key:
        raise RuntimeError('Expected Parrot authentication is unavailable. Use your logged-in EDSL environment or EXPECTED_PARROT_API_KEY.')
    handler = JobsRemoteInferenceHandler(job, api_key=coop.api_key)
    atomic_json({'status': 'submitting'}, state_path)
    info = handler.create_remote_inference_job(fresh=True)
    if not info.job_uuid:
        raise RuntimeError('No remote job ID was returned; collection stopped.')
    state = {'status': 'submitted', 'job_uuid': str(info.job_uuid)}
    atomic_json(state, state_path)
    results, reason = handler.poll_remote_inference_job(info)
    if results is None:
        atomic_json({**state, 'status': 'failed', 'reason': str(reason or 'remote failure')}, state_path)
        raise RuntimeError(f'Remote collection failed: {reason}')
    atomic_csv(results.to_pandas(), raw_path)
    atomic_json({**state, 'status': 'returned'}, state_path)
    return pd.read_csv(raw_path, dtype=str, keep_default_na=False)


def saved_prompt_text(value):
    text = str(value)
    if text.startswith('{'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and 'text' in parsed:
                return parsed['text']
        except (ValueError, SyntaxError):
            pass
    return text


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding=SAVE_ENCODING)
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def household_id(value):
    # Keep text IDs and leading zeroes; never pass household IDs through floats.
    return re.sub(r"^(\d+)\.0+$", r"\1", str(value).strip())


def validate_personas(data, expected_count=None):
    missing = set(PERSONA_FIELDS) - set(data.columns)
    if missing:
        raise ValueError(f"Step 1 persona columns missing: {sorted(missing)}")
    data = data[PERSONA_FIELDS].copy()
    # A missing historical durable expenditure stays unknown, not zero.
    values = data['last_year_durable_spending']
    unknown = values.isna() | values.astype(str).str.strip().isin(['', '未知'])
    amounts = pd.to_numeric(values.where(~unknown), errors='raise')
    observed = amounts.dropna()
    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError('Durable expenditure must be nonnegative and finite, or unknown.')
    data['last_year_durable_spending'] = amounts.map(lambda x: '未知' if pd.isna(x) else format(x, 'g'))
    if data.empty or (expected_count is not None and len(data) != expected_count):
        raise ValueError("Persona sample is empty or does not match --expected-count.")
    for col in TEXT_FIELDS:
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {col}; no households are removed or filled in.")
        data[col] = data[col].astype(str)
    data["hhid"] = data.hhid.map(household_id)
    if data.hhid.duplicated().any():
        raise ValueError("Duplicate hhid values.")
    for col in set(PERSONA_FIELDS) - TEXT_FIELDS:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Nonfinite {col}; use the reviewed Step 1 output.")
    for col in ["agent_idx", "total_population", "seniors_60_or_over", "children_under_14"]:
        if not (data[col].ge(0) & data[col].eq(data[col].astype(int))).all():
            raise ValueError(f"{col} must contain nonnegative integers.")
    if data.agent_idx.duplicated().any():
        raise ValueError("Duplicate agent_idx values.")
    data["agent_idx"] = data.agent_idx.astype(int)
    if not data[WEIGHT_COLUMN].gt(0).all() or not np.isfinite(data[WEIGHT_COLUMN].sum()):
        raise ValueError(f"{WEIGHT_COLUMN} must contain finite positive weights.")
    return data.reset_index(drop=True)


def load_personas(path, expected_count=None):
    # Read only this file. There are no extra CHFS/weight/demographic lookups.
    return validate_personas(pd.read_csv(path, dtype={"hhid": str}), expected_count)


def experiment_specification(data, experiments):
    agents = make_agents(data)
    return {
        "schema_version": 3, "step": 4, "application": APPLICATION,
        "persona_variant": PERSONA_VARIANT,
        "edsl": REQUIRED_EDSL_VERSION, "model": MODEL_NAME,
        "service": MODEL_SERVICE, "model_parameters": MODEL_PARAMETERS,
        "senior_definition": "age >= 60", "weight_column": WEIGHT_COLUMN,
        "experiments": list(experiments),
        "surveys": {f"{exp}/{arm}": make_survey(exp, arm).to_dict()
                    for exp in experiments for arm in ARMS},
        "scenarios": {f"{exp}/{arm}": dict(make_scenario(exp, arm))
                      for exp in experiments for arm in ARMS},
        "personas": [{"traits": dict(agent.traits), "instruction": agent.instruction,
                       "template": agent.traits_presentation_template} for agent in agents],
        "quality_rules": QUALITY_RULES,
        "collection": "single pass; stop on missing or invalid answers",
    }


def actual_prompts(agent, survey, scenario, question, previous_answers):
    from edsl.invigilators.invigilators import InvigilatorAI
    return InvigilatorAI(agent=agent, question=question, scenario=scenario,
                        model=SimpleNamespace(model=MODEL_NAME), survey=survey,
                        memory_plan=survey.memory_plan, current_answers=previous_answers,
                        iteration=0).get_prompts()


def write_prompt_preview(agent, experiments, path):
    lines = ["OFFLINE QUESTIONNAIRE PREVIEW — no model calls and no respondent answers.",
             "First input persona; all <...> values below are placeholders for prior answers.",
             "This file contains private persona information. Keep it out of the public repository."]
    for exp in experiments:
        for arm in ARMS:
            survey, scenario = make_survey(exp, arm), make_scenario(exp, arm)
            previous = {}
            for q in survey.questions:
                prompts = actual_prompts(agent, survey, scenario, q, previous)
                for kind in ("system_prompt", "user_prompt"):
                    lines += [f"\n=== {exp} / {arm} / {q.question_name} / {kind} ===", prompts[kind].text]
                previous[q.question_name] = f"<previous answer: {q.question_name}>"
    Path(path).write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def answer_path(run_dir, experiment, arm):
    return Path(run_dir) / experiment / f"answers_{arm}.csv"


def empty_frames(data, experiments):
    frames = {}
    for exp in experiments:
        for arm in ARMS:
            frame = pd.DataFrame({"agent.hhid": data.hhid, "agent.agent_idx": data.agent_idx.astype(str)})
            for q in question_names(exp):
                frame["answer." + q] = ""
            frames[exp, arm] = frame
    return frames


def pending_ids(frame, experiment):
    return [int(float(row["agent.agent_idx"])) for row in frame.to_dict("records")
            if response_problems(row, experiment)]


def save_status(frames, run_dir, status):
    cells = [{"experiment": exp, "arm": arm, "households": len(frame),
              "pending": len(pending_ids(frame, exp))} for (exp, arm), frame in frames.items()]
    pending = sum(c["pending"] for c in cells)
    atomic_json({"status": "complete" if pending == 0 else status,
                 "pending_questionnaires": pending,
                 "all_complete": pending == 0, "cells": cells}, Path(run_dir) / "status.json")
    return pending


def prepare_run(data, input_path, experiments, output_root, batch_size=None):
    specification = experiment_specification(data, experiments)
    agents = make_agents(data)
    run_dir = Path(output_root) / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    # Freeze the needed traits and weights locally, preserving numeric types.
    atomic_json(data.to_dict("records"), run_dir / "personas.json")
    atomic_json(specification, run_dir / "questionnaire.json")
    write_prompt_preview(agents[0], experiments, run_dir / "prompt_preview.txt")
    frames = empty_frames(data, experiments)
    for (exp, arm), frame in frames.items():
        atomic_csv(frame, answer_path(run_dir, exp, arm))
    atomic_json({"step": 4, "persona_variant": PERSONA_VARIANT,
                 "application": APPLICATION, "experiments": list(experiments),
                 "input_sha256": file_hash(input_path), "script_sha256": file_hash(__file__),
                 "personas_sha256": file_hash(run_dir / "personas.json"),
                 "signature": json_hash(specification), "n_households": len(data),
                 "batch_sizes": {exp: batch_size or default_batch_size(exp) for exp in experiments},
                 "model_calls_at_preparation": 0}, run_dir / "manifest.json")
    save_status(frames, run_dir, "prepared_offline")
    return run_dir


def check_frame_identities(frame, data):
    required = {"agent.hhid", "agent.agent_idx"}
    if required - set(frame) or len(frame) != len(data):
        raise ValueError("Saved answers must retain every input household.")
    ids = frame["agent.hhid"].map(household_id)
    indices = pd.to_numeric(frame["agent.agent_idx"], errors="raise")
    if (ids.duplicated().any() or indices.duplicated().any() or not np.isfinite(indices).all()
            or not indices.eq(indices.astype(int)).all()):
        raise ValueError("Duplicate/invalid saved identities.")
    expected = dict(zip(data.agent_idx, data.hhid))
    if dict(zip(indices.astype(int), ids)) != expected:
        raise ValueError("Saved answer identities differ from the frozen Step 1 sample.")
    return frame.assign(**{"agent.hhid": ids, "agent.agent_idx": indices.astype(int).astype(str)})


def validate_returned_row(row, experiment, arm, agent):
    """Check the exact questionnaire and its own preceding answers, offline."""
    if household_id(row.get("agent.hhid", "")) != agent.traits["hhid"]:
        raise ValueError("Returned hhid differs from the submitted household.")
    if float(row.get("agent.agent_idx", -1)) != agent.traits["agent_idx"]:
        raise ValueError("Returned agent_idx differs from the submitted household.")
    scenario = make_scenario(experiment, arm)
    for key, value in scenario.items():
        if row.get("scenario." + key) != value:
            raise ValueError("Returned policy/scenario differs from the submitted experiment.")
    if (row.get("model.model") != MODEL_NAME or row.get("model.inference_service") != MODEL_SERVICE
            or float(row.get("model.temperature", -1)) != MODEL_TEMPERATURE):
        raise ValueError("Returned model settings differ from the submitted experiment.")
    for key, expected in MODEL_PARAMETERS.items():
        value = row.get("model." + key, "")
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value) if value else None
            except (ValueError, SyntaxError):
                pass
        if value != expected:
            raise ValueError(f"Returned model parameter changed: {key}")
    reasons = response_problems(row, experiment)
    if reasons:
        return reasons
    survey = make_survey(experiment, arm)
    names = question_names(experiment)
    for position, q in enumerate(survey.questions):
        name = q.question_name
        if row.get(f"question_text.{name}_question_text") != q.question_text:
            raise ValueError("Returned question text changed.")
        prompts = {kind: saved_prompt_text(row.get(f"prompt.{name}_{kind}", ""))
                   for kind in ("user_prompt", "system_prompt")}
        if any(not text or text == "NA" for text in prompts.values()):
            raise ValueError("Returned actual prompts are missing.")
        memory = re.findall(r"^\s*Answer:\s*(.*?)\s*$", prompts["user_prompt"], flags=re.MULTILINE)
        if len(memory) != position:
            raise ValueError("Preceding-answer memory has the wrong length.")
        previous = {}
        for prior, text in zip(names[:position], memory):
            if parse_answer(prior, text) != parse_answer(prior, row["answer." + prior]):
                raise ValueError("Later questions remember another questionnaire's answer.")
            try:
                previous[prior] = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                previous[prior] = text
        expected = actual_prompts(agent, survey, scenario, q, previous)
        for kind, text in prompts.items():
            if " ".join(text.split()) != " ".join(expected[kind].text.split()):
                raise ValueError(f"Returned {kind} differs from the requested persona, policy or question.")
    return []


def load_run(run_dir, verify_prompts=True):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get('step') != 4:
        raise ValueError('Expected a run prepared by this Step 4 script.')
    set_persona_variant(manifest.get('persona_variant'))
    if manifest["application"] != APPLICATION:
        raise ValueError("This run belongs to the other application.")
    if file_hash(run_dir / "personas.json") != manifest["personas_sha256"]:
        raise ValueError("Frozen persona file changed.")
    data = validate_personas(pd.DataFrame(json.loads((run_dir / "personas.json").read_text())))
    experiments = manifest["experiments"]
    specification = experiment_specification(data, experiments)
    saved_spec = json.loads((run_dir / "questionnaire.json").read_text())
    if json_hash(specification) != manifest["signature"] or json_hash(saved_spec) != manifest["signature"]:
        raise ValueError("Personas, questions, model settings or quality rules changed; do not mix runs.")
    agents = {int(a.traits["agent_idx"]): a for a in make_agents(data)}
    frames = {}
    for exp in experiments:
        for arm in ARMS:
            frame = pd.read_csv(answer_path(run_dir, exp, arm), dtype=str, keep_default_na=False)
            frame = check_frame_identities(frame, data)
            if verify_prompts:
                for row in frame.to_dict("records"):
                    if not response_problems(row, exp):
                        validate_returned_row(row, exp, arm, agents[int(row["agent.agent_idx"])])
            frames[exp, arm] = frame
    return data, agents, frames, manifest


def build_job(experiment, arm, ids, agents_by_idx, model):
    from edsl.jobs import Jobs
    survey = make_survey(experiment, arm)
    job = survey.by(make_scenario(experiment, arm)).by(model).by(AgentList([agents_by_idx[i] for i in ids]))
    restored = Jobs.from_dict(json.loads(json.dumps(job.to_dict())))
    if restored.survey.to_dict() != survey.to_dict() or len(restored.interviews()) != len(ids):
        raise ValueError("Survey serialization changed the questionnaire or sample size.")
    return restored


def apply_batch(frame, returned, ids, experiment, arm, agents):
    """Validate and store each original complete questionnaire."""
    result = frame.copy()
    if not returned.empty:
        returned_ids = pd.to_numeric(returned["agent.agent_idx"], errors="raise")
        if (not np.isfinite(returned_ids).all() or not returned_ids.eq(returned_ids.astype(int)).all()
                or returned_ids.duplicated().any() or set(returned_ids) - set(ids)):
            raise ValueError("Unexpected or duplicate returned household indices.")
        returned = returned.set_index(returned_ids.astype(int), drop=False)
    positions = {int(float(row["agent.agent_idx"])): i for i, row in result.iterrows()}
    decisions = []
    for idx in ids:
        position = positions[idx]
        if not response_problems(result.loc[position], experiment):
            raise ValueError("This household already has an accepted questionnaire.")
        if returned.empty or idx not in returned.index:
            reasons = ["missing_complete_questionnaire"]
        else:
            row = returned.loc[idx]
            reasons = validate_returned_row(row, experiment, arm, agents[idx])
            if not reasons:
                for col in row.index:
                    if col not in result:
                        result[col] = ""
                result.loc[position, :] = ""
                for col, value in row.items():
                    result.at[position, col] = str(value)
        decisions.append({"agent_idx": idx, "accepted": not reasons, "reason": ";".join(reasons)})
    return result, pd.DataFrame(decisions)


def collect(run_dir, data, agents, frames, manifest,
            runner=run_remote_batch, job_builder=build_job, model=None):
    """One pass through all households; stop if a batch is incomplete or invalid."""
    status = json.loads((Path(run_dir) / 'status.json').read_text())
    if status['status'] != 'prepared_offline':
        raise RuntimeError('Collection has already started for this run; it cannot be submitted again.')
    save_status(frames, run_dir, 'collecting')
    if model is None:
        from edsl import Model
        model = Model(MODEL_NAME, service_name=MODEL_SERVICE, **MODEL_PARAMETERS)
    for (exp, arm), frame in list(frames.items()):
        ids = list(data.agent_idx.astype(int))
        batch_size = manifest['batch_sizes'][exp]
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            directory = Path(run_dir) / 'batches' / f'{exp}_{arm}_{start // batch_size + 1:03d}'
            job = job_builder(exp, arm, batch_ids, agents, model)
            atomic_json(job.to_dict(), directory / 'request.json')
            print(f'{exp}/{arm}: {len(batch_ids)} questionnaires', flush=True)
            returned = runner(job, directory)
            atomic_csv(returned, directory / 'remote_results.csv')
            updated, decisions = apply_batch(frames[exp, arm], returned, batch_ids, exp, arm, agents)
            atomic_csv(updated, answer_path(run_dir, exp, arm))
            atomic_csv(decisions, directory / 'decisions.csv')
            frames[exp, arm] = updated
            if not decisions.accepted.all():
                save_status(frames, run_dir, 'incomplete')
                raise ValueError(f'{exp}/{arm}: missing or invalid answers; collection stopped. See {directory}.')
    save_status(frames, run_dir, 'complete')
    return True


def weighted_inference(values, weights):
    """Household-paired HC1, t(n-1); not a complex-survey-design estimator."""
    x, w = np.asarray(values, float), np.asarray(weights, float)
    if x.ndim != 1 or x.shape != w.shape or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Inference requires at least two complete matched households.")
    if not np.isfinite(w).all() or not (w > 0).all() or not np.isfinite(w.sum()):
        raise ValueError("Invalid household weights.")
    n, a = len(x), w / w.sum()
    estimate = float(a @ x)
    se = float(np.sqrt(n / (n - 1) * np.sum((a * (x - estimate)) ** 2)))
    statistic = estimate / se if se else (0.0 if estimate == 0 else np.copysign(np.inf, estimate))
    margin = float(stats.t.ppf(.975, n - 1) * se)
    return dict(n=n, estimate=estimate, standard_error=se, ci95_lower=estimate-margin,
                ci95_upper=estimate+margin, p_value=float(2 * stats.t.sf(abs(statistic), n - 1)))


def analyze(data, frames, run_dir):
    """ATEs within this persona specification only; no baseline-run dependency."""
    if any(pending_ids(frame, exp) for (exp, arm), frame in frames.items()):
        raise ValueError('Incomplete questionnaires; no ATE is produced and no households are excluded.')
    ordered = data.set_index('hhid')
    weights = ordered[WEIGHT_COLUMN].to_numpy(float)
    cells, estimates = {}, []
    for (exp, arm), frame in frames.items():
        frame = check_frame_identities(frame, data).set_index('agent.hhid').loc[ordered.index]
        cells[exp, arm] = outcome_arrays(frame, exp)
    for exp in dict.fromkeys(exp for exp, arm in frames):
        for (horizon, outcome), control in cells[exp, 'control'].items():
            treatment = cells[exp, 'treatment'][horizon, outcome]
            estimates.append(dict(persona_variant=PERSONA_VARIANT, experiment=exp,
                                  horizon=horizon, outcome=outcome, weighting=WEIGHT_COLUMN,
                                  **weighted_inference(treatment-control, weights)))
    directory = Path(run_dir)/'analysis'/datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(estimates)
    atomic_csv(result, directory/'ate_estimates.csv')
    atomic_json({'step': 4, 'persona_variant': PERSONA_VARIANT,
                 'n_households': len(data), 'weight_column': WEIGHT_COLUMN,
                 'matching_key': 'hhid', 'inference': 'household-paired HC1; t(n-1)',
                 'model_calls': 0,
                 'answer_hashes': {f'{exp}/{arm}': file_hash(answer_path(run_dir, exp, arm))
                                   for exp, arm in frames},
                 'manifest_sha256': file_hash(Path(run_dir)/'manifest.json')},
                directory/'analysis_audit.json')
    print(f'ATE results: {directory}')
    return result, directory


def synthetic_personas():
    """Invented records only; never sampled from CHFS."""
    records = []
    for i in range(2):
        row = {key: 100 for key in PERSONA_FIELDS}
        row.update(hhid=f"SYNTH_{i:03d}", agent_idx=i, province="浙江省", area_type="城市",
                   gender_code=1+i, birth_year=1960+i*10, total_population=3,
                   hh_income=30000+i*10000, total_assets=100000, total_liabilities=20000,
                   net_worth=80000, seniors_60_or_over=1-i, children_under_14=i)
        row.update(owned_durable_text='黑白/彩色电视机、洗衣机',
                   not_owned_durable_text='摄像机/照相机、冰箱、空调、电子计算机/电脑、音响、太阳能/电能热水器、家具、卫星接收器、乐器',
                   last_year_durable_spending=0 if i == 0 else None)
        row[WEIGHT_COLUMN] = float(i+1)
        records.append(row)
    return validate_personas(pd.DataFrame(records))


def synthetic_response(agent, experiment, arm, answers):
    """Construct a fixture with the actual EDSL renderer and invented answers."""
    survey, scenario = make_survey(experiment, arm), make_scenario(experiment, arm)
    row = {"agent.hhid": agent.traits["hhid"], "agent.agent_idx": str(agent.traits["agent_idx"]),
           "model.model": MODEL_NAME, "model.inference_service": MODEL_SERVICE,
           "model.temperature": "1"}
    row.update({"model." + key: "" if value is None else str(value) for key, value in MODEL_PARAMETERS.items()})
    row.update({"scenario." + k: v for k, v in scenario.items()})
    previous = {}
    for q in survey.questions:
        name = q.question_name
        value = answers[name]
        prompts = actual_prompts(agent, survey, scenario, q, previous)
        row.update({"answer." + name: str(value), f"validated.{name}_validated": "True",
                    f"question_text.{name}_question_text": q.question_text,
                    f"raw_model_response.{name}_raw_model_response": repr({"choices": [{"message": {"content": str(value)}}]})})
        for kind in ("system_prompt", "user_prompt"):
            row[f"prompt.{name}_{kind}"] = prompts[kind].text
        previous[name] = value
    return row


def self_test():
    import tempfile
    data = synthetic_personas()
    application_self_test(data)
    with tempfile.TemporaryDirectory(prefix="subsidy_step4_") as temporary:
        directory = Path(temporary)
        source = directory / "invented_personas.csv"
        data.to_csv(source, index=False)
        run_dir = prepare_run(load_personas(source), source, list(EXPERIMENTS), directory)
        data, agents, frames, manifest = load_run(run_dir)
        for (exp, arm), frame in list(frames.items()):
            rows = [synthetic_response(a, exp, arm, example_answers(exp, arm == "treatment", idx*10))
                    for idx, a in agents.items()]
            returned = pd.DataFrame(rows)
            updated, decisions = apply_batch(frame, returned, list(agents), exp, arm, agents)
            assert decisions.accepted.all() and not pending_ids(updated, exp)
            frames[exp, arm] = updated
            atomic_csv(updated, answer_path(run_dir, exp, arm))
            bad = dict(rows[0]); bad["scenario.policy_info"] += "changed"
            try:
                validate_returned_row(bad, exp, arm, agents[0])
            except ValueError:
                pass
            else:
                raise AssertionError("Changed policy was accepted.")
            if len(question_names(exp)) > 1:
                bad = dict(rows[0])
                key = f"prompt.{question_names(exp)[1]}_user_prompt"
                bad[key] = re.sub(r"(^[ \t]*Answer:[ \t]*)[^\n]+", r"\g<1>999", bad[key],
                                  count=1, flags=re.MULTILINE)
                try:
                    validate_returned_row(bad, exp, arm, agents[0])
                except ValueError:
                    pass
                else:
                    raise AssertionError("Wrong preceding-answer memory was accepted.")
        assert save_status(frames, run_dir, "complete") == 0
        _, _, reloaded, _ = load_run(run_dir)
        estimates, _ = analyze(data, reloaded, run_dir)
        assert len(estimates) and (estimates.n == 2).all()
        assert (estimates.estimate >= 0).all()
        wrong = data.copy(); wrong.loc[1, "hhid"] = wrong.loc[0, "hhid"]
        try:
            validate_personas(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate household ID was accepted.")
        # Explicit counterexample to row-order pairing and equal-weight mistakes.
        inferred = weighted_inference([10, 40], [1, 2])
        assert np.isclose(inferred["estimate"], 30)
        assert np.isclose(inferred["standard_error"], 40/3)
    print(f"{APPLICATION}/{PERSONA_VARIANT}: offline synthetic checks passed; model calls = 0.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "collect", "analyze"], default="prepare",
                        help="Default prepare is offline. Only collect can submit paid requests.")
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "outputs/step1/personas.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs/step4")
    parser.add_argument("--run-dir", type=Path, help="A prepared or completed run from this Step 4 script.")
    parser.add_argument("--variant", choices=VARIANTS, help=f"New runs only; default: {VARIANTS[0]}.", default=None)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=None)
    parser.add_argument("--expected-count", type=int, help="Optional sample-size check; never pads/truncates.")
    parser.add_argument("--batch-size", type=int, help="New runs only; otherwise keep saved batch sizes.")
    parser.add_argument("--self-test", action="store_true", help="Invented personas/answers, no network calls.")
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("Use positive --batch-size.")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive.")
    if args.experiments and len(set(args.experiments)) != len(args.experiments):
        parser.error("Duplicate experiment names.")
    if args.mode == "analyze" and args.run_dir is None:
        parser.error("--mode analyze requires --run-dir.")
    if args.run_dir is not None and (args.experiments is not None or args.batch_size is not None or args.variant is not None):
        parser.error("An existing Step 4 run uses its saved variant, experiments and batch sizes.")
    load_edsl()
    if args.self_test:
        for variant in VARIANTS:
            set_persona_variant(variant)
            self_test()
        return 0
    if args.run_dir is None:
        set_persona_variant(args.variant or VARIANTS[0])
        data = load_personas(args.input, args.expected_count)
        args.run_dir = prepare_run(data, args.input, args.experiments or list(EXPERIMENTS),
                                   args.output_dir / PERSONA_VARIANT, args.batch_size)
    # Prevent concurrent collection of the same run.
    import fcntl
    with (args.run_dir / ".step4.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, agents, frames, manifest = load_run(args.run_dir)
        if args.expected_count is not None and len(data) != args.expected_count:
            raise ValueError("Saved run does not match --expected-count.")
        print(f"Run: {args.run_dir}")
        if args.mode == "prepare":
            print("Offline preparation/check complete. No model requests submitted.")
            return 0
        if args.mode == "collect":
            pending = any(pending_ids(frame, exp) for (exp, arm), frame in frames.items())
            if pending:
                collect(args.run_dir, data, agents, frames, manifest)
        analyze(data, frames, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
