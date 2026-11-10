"""Step 4: persona sensitivity, within-specification ATEs only.

Two variants: no_consumption removes past consumption; no_persona removes all
household information from the prompt. Each category/horizon is an independent
cumulative spending question, as in the reviewed Appendix-aligned questionnaire.
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


APPLICATION = "digital_coupon"
WEIGHT_COLUMN = "weight_hh"
FIRST_7D_OUTCOME = "personal_alipay_total_oop_7d"


TOTAL_6W_OUTCOME = "personal_alipay_total_oop_6w"


TOTAL_20W_OUTCOME = "personal_alipay_total_oop_20w"

CATEGORY_CONFIGS = {
    'dining': {
        "display_name": "餐饮",
        "policy_name": "餐饮",
        "packet_name": "餐饮券包",
        "coupon_count": 2,
        "coupon_count_text": "两",
        "coupon_list": """
1. 一张满90元减30元餐饮券；
2. 一张满210元减70元餐饮券。
""",
        "eligibility_text": """
每张券只有在符合条件的线下餐饮商户发生一笔达到相应最低门槛的
真实消费时才可使用，并减免相应金额。餐饮券只适用于被平台归入
餐饮类别的合资格线下商户。不要自行添加其他服务项目或商品限制。
""",
        "scope_text": "餐饮",
    },
    'books_digital': {
        "display_name": "图书＋手机及配件",
        "policy_name": "图书＋手机及配件",
        "packet_name": "图书＋手机及配件券包",
        "coupon_count": 3,
        "coupon_count_text": "三",
        "coupon_list": """
图书消费券：
1. 第一张满50元减25元图书券；
2. 第二张满50元减25元图书券。两张图书券彼此独立，分别需要一笔
   达到或超过50元的合资格图书类线下交易才能核销。

手机及配件消费券：
3. 一张满2000元减200元手机及配件券。
""",
        "eligibility_text": """
每张券只有在对应类别的合资格线下商户发生一笔达到相应最低门槛的
真实消费时才可使用，并减免相应金额。图书券只适用于被平台归入
图书类别的合资格线下商户；手机及配件券只适用于被平台归入
手机及配件类别的合资格线下商户。不要自行添加其他商品、品牌
或品类限制。
""",
        "scope_text": "图书或手机及配件",
    },
}

VARIANTS = ('no_consumption', 'no_persona')
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
    except (TypeError, ValueError):
        return "性别信息未知"

    if value == 1:
        return "男性"
    if value == 2:
        return "女性"
    return "性别信息未知"


def describe_age_in_2020(value):
    try:
        birth_year = int(float(value))
    except (TypeError, ValueError):
        return "年龄信息未知"

    if birth_year <= 1900 or birth_year > 2020:
        return "年龄信息未知"
    return f"2020年时{2020 - birth_year}岁"


COMMON_SURVEY_CONTEXT = '\n当前是2020年4月。浙江省城市已逐步从新冠疫情管控中恢复。\n请参考2020年4月当时的物价、收入预期和线下消费环境，\n不要使用2020年4月之后发生的信息，以及任何外部知识或现实世界的数据。\n\n本调查的主要结果变量以你本人实名认证的支付宝账户为单位。\n请假定你平时会使用支付宝进行线下日常消费。\n请以自己的实际情况和消费意愿出发，回答后续问题。\n'


def add_traits_presentation(agent_list):
    if PERSONA_VARIANT == 'no_persona':
        context = COMMON_SURVEY_CONTEXT
        return AgentList([Agent(traits=agent.traits, traits_presentation_template=context)
                          for agent in agent_list])
    new_agents = []

    for agent in agent_list:
        seniors_count = clean_count(
            agent.traits.get("seniors_60_or_over", 0)
        )
        children_count = clean_count(
            agent.traits.get("children_under_14", 0)
        )
        seniors_text = describe_seniors(seniors_count)
        children_text = describe_children(children_count)
        gender_text = describe_gender(agent.traits.get("gender_code", 0))
        age_text = describe_age_in_2020(agent.traits.get("birth_year", 0))

        new_agent = Agent(
            traits=agent.traits,
            traits_presentation_template=f"""
你是居住在中国{agent.traits.get("province", "未知省份")}的一位
{agent.traits.get("area_type", "未知地区")}户主。

你的性别为{gender_text}，{age_text}。

你的家庭共有{agent.traits.get("total_population", "未知")}人。
你的家庭年收入为{agent.traits.get("hh_income", "未知")}元。
你的家庭总资产为{agent.traits.get("total_assets", "未知")}元，
家庭总负债为{agent.traits.get("total_liabilities", "未知")}元，
家庭净资产为{agent.traits.get("net_worth", "未知")}元。

你的家庭中{seniors_text}，并且{children_text}。

当前是2020年4月。浙江省城市已逐步从新冠疫情管控中恢复。
请参考2020年4月当时的物价、收入预期和线下消费环境，
不要使用2020年4月之后发生的信息，以及任何外部知识或现实世界的数据。

本调查的主要结果变量以你本人实名认证的支付宝账户为单位。
请假定你平时会使用支付宝进行线下日常消费。
请以自己的实际情况和消费意愿出发，回答后续问题。
""",
        )
        new_agents.append(new_agent)

    return AgentList(new_agents)


def build_common_policy(config):
    return f"""
你所在的城市位于浙江省，正在通过支付宝发放{config['policy_name']}消费券。

本轮这一组消费券包含以下{config['coupon_count_text']}张券：
{config['coupon_list'].strip()}

{config['eligibility_text'].strip()}

消费券规则如下：

- 仅限消费券对应类别的合资格线下商户使用；
- 不能用于线上交易；
- 如果一笔消费没有达到相应的最低消费门槛，
  则不能使用对应的消费券；
- 符合使用条件时，消费券在支付宝付款环节自动抵扣；
- 消费券不能提现、不能转让；
- 每张消费券最多核销一次，并且只能用于一笔交易。

领取当天记为第1天。

这{config['coupon_count_text']}张消费券是本轮一次性发放的支付优惠：
- 只能在第1天至第7天内符合条件的交易中核销；
- 到第7天结束时，未核销的消费券都会自动失效；
- 从第8天开始，这{config['coupon_count_text']}张消费券不再出现在付款抵扣选项中，
  也不会转化为支付宝余额、现金、积分或其他可在以后使用的权益；
- 本轮消费券不会续期、刷新或补发。

本轮{config['policy_name']}消费券数量有限，采用先到先得的方式发放。领取入口在上午10点开放，用户需要主动进入领取页面，
并按照首次进入页面的先后顺序领取。

在进入领取页面前，你已经看到了这{config['coupon_count_text']}张消费券的面额、
最低消费门槛、适用商户和7天有效期。
在了解这些规则后，你根据自己的实际情况，
主动进入领取页面参加了本轮抢券。

领取入口开放后，大量用户同时进入页面抢券。
本轮{config['packet_name']}在入口开放约两分钟后全部被领完。
在最后一份{config['packet_name']}被领完之前首次进入页面的用户，
按照先到先得的顺序成功获得券包；
此后首次进入页面的用户无法获得本轮券包。
"""


def build_control_policy(config, common_policy):
    return (
        common_policy
        + f"""
你也在领取入口开放后的前约两分钟内进入页面，
但你的首次进入时间略晚于最后一份{config['packet_name']}被领完的时点。
当页面加载完成时，本轮{config['packet_name']}刚刚被全部领完。

因此，你本人本轮没有获得上述{config['coupon_count_text']}张消费券。
在未来7天内，你也不会获得其他新的{config['policy_name']}消费券。
"""
    )


def build_treatment_policy(config, common_policy):
    return (
        common_policy
        + f"""
你在领取入口开放后的前约两分钟内进入页面，
你的首次进入时间略早于最后一份{config['packet_name']}被领完的时点。
因此，你成功获得了一个{config['packet_name']}，
其中包含上述{config['coupon_count_text']}张消费券。

这{config['coupon_count_text']}张消费券已经进入你本人实名认证的支付宝账户，
你可以在未来7天内按照上述规则使用。
"""
    )

def make_first_7d_question(category_key):
    config = CATEGORY_CONFIGS[category_key]
    return QuestionNumerical(
        question_name=FIRST_7D_OUTCOME,
        question_text=f"""
{{{{ policy_info }}}}

在上述情况下，未来连续7天内，
你预计通过本人实名认证支付宝账户进行的所有消费，
其实际自付总金额是多少元？

这里的“所有消费”是指通过该支付宝账户购买商品或服务的全部消费，
不限于{config['scope_text']}消费，也不限于可以使用上述消费券的交易。

""",
        include_comment=False,
        answering_instructions="""
只返回一个数字。
不要添加“元”、逗号、解释或其他文字。
""",
    )


HORIZONS = {
    "6w": {"outcome": TOTAL_6W_OUTCOME, "period_text": "第1天至第42天（前6周，共42天）"},
    "20w": {"outcome": TOTAL_20W_OUTCOME, "period_text": "第1天至第140天（前20周，共140天）"},
}


def make_horizon_question(
    category_key,
    arm,
    outcome_name,
    period_text,
):
    config = CATEGORY_CONFIGS[category_key]

    if arm == "control":
        later_period_policy_text = f"""
在这一期间内，不考虑获得其他新的{config['policy_name']}消费券。
"""
    elif arm == "treatment":
        later_period_policy_text = f"""
从第8天起，这{config['coupon_count_text']}张消费券都已经超过7天有效期并失效，
不能再用于任何交易，也不能再抵扣任何金额。
在这一期间内，不考虑获得其他新的{config['policy_name']}消费券。
"""
    else:
        raise ValueError(f"未知实验组别：{arm}")

    return QuestionNumerical(
        question_name=outcome_name,
        question_text=f"""
{{{{ policy_info }}}}

现在只考虑未来{period_text}。

{later_period_policy_text.strip()}

在这种情况下，你预计在这期间，
通过本人实名认证支付宝账户进行的所有消费，
其累计实际自付总金额是多少元？

这里的“所有消费”是指通过该支付宝账户购买商品或服务的全部消费，
不限于{config['scope_text']}消费。

""",
        include_comment=False,
        answering_instructions="""
只返回一个数字。
不要添加“元”、逗号、解释或其他文字。
""",
    )

EXPERIMENTS = {
    "dining_7d": ("dining", "7d"),
    "books_digital_7d": ("books_digital", "7d"),
    "dining_6w": ("dining", "6w"),
    "dining_20w": ("dining", "20w"),
    "books_digital_6w": ("books_digital", "6w"),
    "books_digital_20w": ("books_digital", "20w"),
}
PERSONA_FIELDS = [
    "hhid", "agent_idx", "province", "area_type", "gender_code", "birth_year",
    "total_population", "hh_income", "total_assets",
    "total_liabilities", "net_worth", "seniors_60_or_over", "children_under_14",
    "weight_hh",
]
TEXT_FIELDS = {"hhid", "province", "area_type"}
COMMON_POLICIES = {k: build_common_policy(v) for k, v in CATEGORY_CONFIGS.items()}
CONTROL_POLICY_INFO = {k: build_control_policy(CATEGORY_CONFIGS[k], v)
                       for k, v in COMMON_POLICIES.items()}
TREATMENT_POLICY_INFO = {k: build_treatment_policy(CATEGORY_CONFIGS[k], v)
                         for k, v in COMMON_POLICIES.items()}


def make_agents(data):
    # Unique bookkeeping identities remain distinct but are never shown in the prompt.
    # Deleted information is not attached to the EDSL agent, including as hidden traits.
    fields = (['hhid', 'agent_idx'] if PERSONA_VARIANT == 'no_persona'
              else [field for field in PERSONA_FIELDS if field != WEIGHT_COLUMN])
    return add_traits_presentation(AgentList([Agent(traits=row)
                                             for row in data[fields].to_dict('records')]))


def make_scenario(experiment, arm):
    category, horizon = EXPERIMENTS[experiment]
    if horizon == "7d":
        name = f"{arm.title()}_{category}_" + ("NoWin" if arm == "control" else "Win")
    else:
        name = f"{arm.title()}_{category}_{horizon}_" + ("NoWin" if arm == "control" else "Win")
    return Scenario({"treatment_name": name, "policy_info":
                     (CONTROL_POLICY_INFO if arm == "control" else TREATMENT_POLICY_INFO)[category]})


def make_survey(experiment, arm):
    category, horizon = EXPERIMENTS[experiment]
    if horizon == "7d":
        question = make_first_7d_question(category)
    else:
        specification = HORIZONS[horizon]
        question = make_horizon_question(category, arm, specification["outcome"],
                                         specification["period_text"])
    # Appendix A: one independent cumulative question for each horizon.
    return Survey(questions=[question])


def question_names(experiment):
    horizon = EXPERIMENTS[experiment][1]
    return [FIRST_7D_OUTCOME if horizon == "7d" else HORIZONS[horizon]["outcome"]]


def parse_answer(question, value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def response_problems(row, experiment):
    # Appendix A specifies a number without amount bounds. Do not retain the
    # historical >=1 rule, per-question caps, or cumulative-cap filters.
    return ["missing_or_nonfinite_number:" + q for q in question_names(experiment)
            if parse_answer(q, row.get("answer." + q)) is None]


def default_batch_size(experiment):
    return 100 if EXPERIMENTS[experiment][1] == "7d" else 200


def outcome_arrays(frame, experiment):
    horizon = EXPERIMENTS[experiment][1]
    q = FIRST_7D_OUTCOME if horizon == "7d" else HORIZONS[horizon]["outcome"]
    return {(horizon, "spending"): pd.to_numeric(frame["answer." + q]).to_numpy(float)}


def example_answers(experiment, treatment=False, offset=0):
    return {q: 100 + offset + 20 * i + (10 if treatment else 0)
            for i, q in enumerate(question_names(experiment))}


def application_self_test(data):
    for experiment in EXPERIMENTS:
        survey = make_survey(experiment, "control")
        assert len(survey.questions) == 1
        question = survey.questions[0]
        assert not survey.memory_plan
        assert question.min_value is None and question.max_value is None
        column = "answer." + question.question_name
        # Zero and amounts above the historical caps must remain admissible.
        for amount in [0, 0.5, 1_000_000]:
            assert not response_problems({column: amount}, experiment)
        for invalid in [None, "", "nan", "inf", "10元"]:
            assert response_problems({column: invalid}, experiment)
    template = make_agents(data)[0].traits_presentation_template
    assert all(phrase not in template for phrase in ['家庭全年总消费', '伙食费', '文化娱乐', '购买及修理通信工具'])
    if PERSONA_VARIANT == 'no_consumption':
        assert '60岁及以上' in template and '你的家庭年收入' in template
    else:
        assert '你的家庭' not in template and '你的性别' not in template
        agents = make_agents(data)
        assert all(a.traits_presentation_template == template for a in agents)
        assert all(set(a.traits) == {'hhid', 'agent_idx'} for a in agents)
    assert "任何外部知识或现实世界的数据" in template
    assert "旅游" not in template and "travel_spending" not in template


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
        "prompt_basis": "submitted Appendix A.2-A.5.1; aligned 2026-11-08",
        "quality_rules": {"require_finite_number": True,
                          "amount_bounds": None, "cross_horizon_checks": False},
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
