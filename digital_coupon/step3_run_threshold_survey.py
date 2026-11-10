"""Step 3: single-transaction coupon threshold survey, Appendix A.5.1.
Self-contained persona, policy, questions, collection,
weighted full-sample histograms and paired threshold-bin inference.
Default prepare is offline. Only --mode collect enables paid surveys.
Input: Step 1 personas.csv. Optional --xing-csv uses published aggregate bars.
Historical scripts and answers are never read implicitly or overwritten.
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

import os
from importlib.metadata import version
from types import SimpleNamespace

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
    global Agent, AgentList, QuestionNumerical, QuestionMultipleChoice, Scenario, ScenarioList, Survey
    if version("edsl") != REQUIRED_EDSL_VERSION:
        raise RuntimeError("Use edsl==1.0.8 to preserve the questionnaire rendering.")
    from edsl import Agent, AgentList, QuestionNumerical, QuestionMultipleChoice, Scenario, ScenarioList, Survey

APPLICATION="digital_coupon_threshold"
WEIGHT_COLUMN="weight_hh"
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
        "xing_benchmarks": {
            FIRST_7D_OUTCOME: 195.79,
            TOTAL_6W_OUTCOME: 164.95,
            TOTAL_20W_OUTCOME: 163.37,
        },
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
        "xing_benchmarks": {
            FIRST_7D_OUTCOME: 948.62,
            TOTAL_6W_OUTCOME: 899.82,
            TOTAL_20W_OUTCOME: 285.62,
        },
    },
}

PERSONA_FIELDS = [
    "hhid", "agent_idx", "province", "area_type", "gender_code", "birth_year",
    "total_population", "hh_income", "annual_consumption", "total_assets",
    "total_liabilities", "net_worth", "seniors_60_or_over", "children_under_14",
    "food_spending", "entertainment_spending", "tele_spending", "weight_hh",
]

TEXT_FIELDS = {"hhid", "province", "area_type"}

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

def add_traits_presentation(agent_list):
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
你的家庭全年总消费为{agent.traits.get("annual_consumption", "未知")}元。
你的家庭总资产为{agent.traits.get("total_assets", "未知")}元，
家庭总负债为{agent.traits.get("total_liabilities", "未知")}元，
家庭净资产为{agent.traits.get("net_worth", "未知")}元。

你的家庭中{seniors_text}，并且{children_text}。

去年，你的家庭平均每月伙食费支出为
{agent.traits.get("food_spending", "未知")}元，
平均每月文化娱乐支出为
{agent.traits.get("entertainment_spending", "未知")}元，
全年购买及修理通信工具（包括手机、车载导航仪等）的支出为
{agent.traits.get("tele_spending", "未知")}元。

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

def make_agents(data):
    return add_traits_presentation(AgentList([Agent(traits=row) for row in data.to_dict("records")]))

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

COMMON_POLICIES = {k: build_common_policy(v) for k, v in CATEGORY_CONFIGS.items()}

CONTROL_POLICY_INFO = {k: build_control_policy(CATEGORY_CONFIGS[k], v)
                       for k, v in COMMON_POLICIES.items()}

TREATMENT_POLICY_INFO = {k: build_treatment_policy(CATEGORY_CONFIGS[k], v)
                         for k, v in COMMON_POLICIES.items()}

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

def actual_prompts(agent, survey, scenario, question, previous_answers):
    from edsl.invigilators.invigilators import InvigilatorAI
    return InvigilatorAI(agent=agent, question=question, scenario=scenario,
                        model=SimpleNamespace(model=MODEL_NAME), survey=survey,
                        memory_plan=survey.memory_plan, current_answers=previous_answers,
                        iteration=0).get_prompts()

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

EXPERIMENT_CONFIGS = {
    "dining": {
        "display_name": "餐饮", "english_name": "Dining",
        "coupons": [(90, 30), (210, 70)], "focus_max": 400,
    },
    "books": {
        "display_name": "图书", "english_name": "Books",
        # 两张同面额券在单笔交易中的可选优惠相同，不人为区分第一/第二张。
        "coupons": [(50, 25)], "focus_max": 200,
    },
    "digital": {
        "display_name": "手机及配件", "english_name": "Cellphones and accessories",
        "coupons": [(2000, 200)], "focus_max": 4000,
    },
}

POLICY_CATEGORY_MAP = {
    "dining": "dining",
    "books": "books_digital",
    "digital": "books_digital",
}

USE_QUESTION_NAME = "coupon_used_for_this_transaction"

HISTOGRAM_BIN_WIDTH = 10

BATCH_SIZE = 100

def bill_question_name(experiment):
    return f"{experiment}_bill_before_discount_yuan"

def coupon_options(experiment):
    config = EXPERIMENT_CONFIGS[experiment]
    name = config["display_name"]
    choices = [f"这笔交易不使用{name}消费券"]
    for threshold, subsidy in config["coupons"]:
        prefix = "使用其中一张" if experiment == "books" else "使用"
        choices.append(f"{prefix}满{threshold}元减{subsidy}元{name}券")
    return choices

def coupon_amounts(experiment):
    return dict(zip(
        coupon_options(experiment),
        [0] + [subsidy for _, subsidy in EXPERIMENT_CONFIGS[experiment]["coupons"]],
    ))

def make_policy_info(experiment, arm):
    policy_category = POLICY_CATEGORY_MAP[experiment]
    if arm == "control":
        policy_info = CONTROL_POLICY_INFO[policy_category]
    elif arm == "treatment":
        policy_info = TREATMENT_POLICY_INFO[policy_category]
    else:
        raise ValueError(f"未知实验组：{arm}")
    name = EXPERIMENT_CONFIGS[experiment]["display_name"]
    # 政策段落原封不动；只在其后接回门槛问卷既有的单笔交易情景。
    return policy_info + f"""
请设想在未来7天内，你已决定在一家线下{name}商户进行一笔{name}消费。
这笔消费会发生。
"""

def make_survey(experiment, arm):
    name = EXPERIMENT_CONFIGS[experiment]["display_name"]
    bill_question = QuestionNumerical(
        question_name=bill_question_name(experiment),
        question_text=f"""
{{{{ policy_info }}}}

这笔{name}交易在任何优惠抵扣前的账单总额最可能是多少元？
""",
        include_comment=False,
        answering_instructions="请直接填写精确金额，只返回一个数字，不要添加单位或解释。",
    )
    if arm == "control":
        return Survey(questions=[bill_question])

    return Survey(questions=[bill_question, make_use_question(experiment)]).set_full_memory_mode()

def make_use_question(experiment):
    name = EXPERIMENT_CONFIGS[experiment]["display_name"]
    return QuestionMultipleChoice(
        question_name=USE_QUESTION_NAME,
        question_text=f"""
对于上一题的这笔{name}交易，你最可能选择如何使用刚才抢到的消费券？
只有账单达到相应门槛的券才可以使用；同一笔交易最多使用一张。
""",
        question_options=coupon_options(experiment),
        include_comment=False,
        use_code=False,
        answering_instructions=(
            "请只返回所选选项的完整文字，"
            "必须与上述某一个选项完全一致。"
            "不要只返回账单金额、门槛金额、优惠金额或选项编号，不要添加解释。"
        ),
    )

def make_scenario(experiment, arm):
    return Scenario({"arm": arm, "policy_info": make_policy_info(experiment, arm)})

def remote_job(job, batch_dir):
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


def parse_complete_answer(row, arm, experiment):
    bill = parse_bill(row.get(bill_question_name(experiment)))
    if bill is None:
        return None
    if arm == "control":
        return bill, None
    choice = row.get(USE_QUESTION_NAME)
    if choice not in coupon_amounts(experiment):
        return None
    # 核对所选消费券的使用门槛。
    subsidy = coupon_amounts(experiment)[choice]
    if subsidy and any(
        subsidy == coupon_subsidy and bill < threshold
        for threshold, coupon_subsidy in EXPERIMENT_CONFIGS[experiment]["coupons"]
    ):
        return None
    return bill, choice

def chunk_agentlist(agent_list):
    for start in range(0, len(agent_list), BATCH_SIZE):
        yield AgentList(agent_list[start:start + BATCH_SIZE])

def safe_file_component(value):
    return str(value).replace("/", "_")

def remove_prefix(column):
    for prefix in (
        "answer.", "answer_", "agent.", "agent_", "scenario.", "scenario_"
    ):
        if column.startswith(prefix):
            return column[len(prefix):]
    return column

def batch_filename(arm, model_name, batch_num, output_dir):
    return os.path.join(
        output_dir,
        f"results_{arm}_{safe_file_component(model_name)}_batch{batch_num:03d}.csv",
    )


def run_arm_batches(arm, survey, scenario, model, agents, experiment, output_dir):
    for batch_num, batch_agents in enumerate(chunk_agentlist(agents), start=1):
        filename = batch_filename(arm, model.model, batch_num, output_dir)
        if Path(filename).exists():
            raise RuntimeError(f'Batch already exists; collection stopped: {filename}')
        results = run_remote_batch(survey, scenario, model, batch_agents, filename)
        atomic_csv(results, Path(filename))
        answers = batch_answers(results, arm, model.model, experiment)
        expected = {int(a.traits['agent_idx']) for a in batch_agents}
        if set(answers.agent_idx) != expected:
            raise ValueError(f'Missing answers; collection stopped. Original results: {filename}')
        print(f'已保存：{filename}')


def batch_answers(frame, arm, model_name, experiment):
    """Validate original bill/choice answers together; do not discard households."""
    frame = frame.rename(columns=remove_prefix)
    if frame.empty or 'agent_idx' not in frame:
        raise ValueError('No household answers returned; collection stopped.')
    ids = pd.to_numeric(frame.agent_idx, errors='raise')
    if not np.isfinite(ids).all() or not ids.eq(ids.astype(int)).all() or ids.duplicated().any():
        raise ValueError('Invalid or duplicate household indices.')
    records = []
    for idx, row in zip(ids.astype(int), frame.to_dict('records')):
        answer = parse_complete_answer(row, arm, experiment)
        if answer is None:
            raise ValueError(f'{experiment}/{arm}: invalid or missing answer for agent_idx={idx}; collection stopped.')
        bill, choice = answer
        records.append(dict(agent_idx=idx, arm=arm, model=model_name,
                            bill_before_discount_yuan=bill,
                            coupon_choice=choice if choice is not None else 'not_asked'))
    return pd.DataFrame(records)

def combine_arm_answers(arm, model_name, experiment, output_dir):
    prefix = f'results_{arm}_{safe_file_component(model_name)}_batch'
    files = sorted(p for p in Path(output_dir).glob(prefix + '*.csv')
                   if re.fullmatch(re.escape(prefix) + r'\d+\.csv', p.name))
    if not files:
        raise FileNotFoundError(f'No original batches in {output_dir}')
    answers = pd.concat([batch_answers(pd.read_csv(p, dtype=str, keep_default_na=False),
                                      arm, model_name, experiment) for p in files], ignore_index=True)
    if answers.agent_idx.duplicated().any():
        raise ValueError('Duplicate households across batches.')
    answers = answers.sort_values('agent_idx').reset_index(drop=True)
    filename = Path(output_dir) / f'answers_{arm}_{safe_file_component(model_name)}.csv'
    atomic_csv(answers, filename)
    print(f'已合并：{filename}，有效样本数={len(answers)}')
    return answers

def match_persona_answers(answers):
    """按agent_idx一对一匹配两组；保存同一persona的账单变化。"""
    control = answers["control"][[
        "agent_idx", "bill_before_discount_yuan",
    ]].rename(columns={
        "bill_before_discount_yuan": "control_bill_before_discount_yuan",
    })
    treatment = answers["treatment"][[
        "agent_idx", "bill_before_discount_yuan", "coupon_choice",
    ]].rename(columns={
        "bill_before_discount_yuan": "treatment_bill_before_discount_yuan",
        "coupon_choice": "treatment_coupon_choice",
    })
    matched = control.merge(
        treatment, on="agent_idx", how="outer", validate="one_to_one",
        indicator=True,
    )
    if not (matched["_merge"] == "both").all():
        raise ValueError("两组persona不完全一致；不生成不完整的配对结果。")
    matched = matched.drop(columns="_merge").sort_values("agent_idx")
    matched["bill_difference_yuan"] = (
        matched["treatment_bill_before_discount_yuan"]
        - matched["control_bill_before_discount_yuan"]
    )
    return matched.reset_index(drop=True)

def add_outcomes(answers, experiment):
    frame = pd.concat(answers.values(), ignore_index=True)
    frame["experiment"] = experiment
    coupons = EXPERIMENT_CONFIGS[experiment]["coupons"]
    bill = frame["bill_before_discount_yuan"]
    for threshold, subsidy in coupons:
        frame[f"eligible_{subsidy}"] = (bill >= threshold).astype(int)
    frame["selected_coupon_yuan"] = frame["coupon_choice"].map(
        coupon_amounts(experiment)
    ).fillna(0).astype(int)
    consistent = frame["selected_coupon_yuan"] == 0
    for threshold, subsidy in coupons:
        consistent |= (frame["selected_coupon_yuan"] == subsidy) & (bill >= threshold)
    frame["coupon_choice_consistent"] = consistent
    frame["subsidized_transaction"] = (
        (frame["selected_coupon_yuan"] > 0)
        & frame["coupon_choice_consistent"]
    ).astype(int)
    frame["actual_subsidy_yuan"] = np.where(
        frame["subsidized_transaction"] == 1,
        frame["selected_coupon_yuan"],
        0,
    )
    frame["oop_after_discount_yuan"] = bill - frame["actual_subsidy_yuan"]
    frame["amount_bin_lower_yuan"] = (
        np.floor(bill / HISTOGRAM_BIN_WIDTH) * HISTOGRAM_BIN_WIDTH
    ).astype(int)
    frame["amount_bin_upper_yuan"] = (
        frame["amount_bin_lower_yuan"] + HISTOGRAM_BIN_WIDTH
    )
    return frame

CONFIGS = {
    'dining': {'name': 'Dining', 'width': 10, 'cutoff': 240,
               'thresholds': [90, 210], 'subsidies': [30, 70],
               'ticks': [0, 50, 100, 150, 200, 240],
               'coupon_text': 'RMB 30 off 90; RMB 70 off 210'},
    'books': {'name': 'Books', 'width': 5, 'cutoff': 100,
              'thresholds': [50], 'subsidies': [25],
              'ticks': [0, 25, 50, 75, 100],
              'coupon_text': 'RMB 25 off 50'},
    'digital': {'name': 'Cellphones and accessories', 'width': 200,
                'cutoff': 3000, 'thresholds': [2000], 'subsidies': [200],
                'ticks': [0, 1000, 2000, 3000],
                'coupon_text': 'RMB 200 off 2,000'},
}

def parse_bill(value):
    try:
        amount = float(value)
        return amount if np.isfinite(amount) and amount >= 0 else None
    except (TypeError, ValueError):
        return None


def prepare(data, source, output, categories, batch_size):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(data.to_dict('records'), output/'personas.json')
    agents = make_agents(data)
    spec = {'version': 'step3_appendix_a_single_pass_20261109', 'model': MODEL_NAME,
            'model_parameters': MODEL_PARAMETERS, 'categories': categories, 'batch_size': batch_size,
            'personas': [{'traits':dict(a.traits),'template':a.traits_presentation_template,'instruction':a.instruction} for a in agents],
            'surveys': {f'{e}/{a}':make_survey(e,a).to_dict() for e in categories for a in ARMS},
            'scenarios': {f'{e}/{a}':dict(make_scenario(e,a)) for e in categories for a in ARMS},
            'quality_rules': 'finite nonnegative bill; valid coupon choice and threshold; single pass; stop on missing or invalid answers'}
    atomic_json(spec, output/'questionnaire.json')
    atomic_json({'signature':json_hash(spec),'input_sha256':file_hash(source),
                 'personas_sha256':file_hash(output/'personas.json'), 'script_sha256':file_hash(__file__),
                 'model_calls_at_preparation':0}, output/'manifest.json')
    write_preview(agents[0], categories, output/'prompt_preview.txt')
    for e in categories: (output/e).mkdir()
    return output


def write_preview(agent, categories, path):
    lines = ['OFFLINE PREVIEW: persona values are from the supplied input; keep private.']
    for e in categories:
        for arm in ARMS:
            survey, scenario, previous = make_survey(e,arm), make_scenario(e,arm), {}
            for q in survey.questions:
                for role,prompt in actual_prompts(agent,survey,scenario,q,previous).items():
                    if role in ['system_prompt','user_prompt']:
                        lines += [f'\n=== {e}/{arm}/{q.question_name}/{role} ===',prompt.text]
                previous[q.question_name] = '<previous bill answer>'
    Path(path).write_text('\n\n'.join(lines),encoding='utf-8')


def load_run(path):
    data = validate_personas(pd.DataFrame(json.loads((path/'personas.json').read_text())))
    spec = json.loads((path/'questionnaire.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['signature'] != json_hash(spec) or manifest['personas_sha256'] != file_hash(path/'personas.json'):
        raise ValueError('Frozen run was modified.')
    agents = make_agents(data)
    if spec['version'] != 'step3_appendix_a_single_pass_20261109' or spec['model_parameters'] != MODEL_PARAMETERS:
        raise ValueError('Different experiment version.')
    if spec['personas'] != [{'traits':dict(a.traits),'template':a.traits_presentation_template,'instruction':a.instruction} for a in agents]:
        raise ValueError('Persona definition changed.')
    for e in spec['categories']:
        for a in ARMS:
            if spec['surveys'][e+'/'+a] != make_survey(e,a).to_dict() or spec['scenarios'][e+'/'+a] != dict(make_scenario(e,a)):
                raise ValueError('Questionnaire or policy changed; do not mix runs.')
    return data,agents,spec


def run_remote_batch(survey, scenario, model, agents, save_filename):
    job = survey.by(scenario).by(model).by(agents)
    directory = Path(str(save_filename)+'.job')
    directory.mkdir(exist_ok=True)
    if not (directory/'request.json').exists(): atomic_json(job.to_dict(),directory/'request.json')
    result = remote_job(job,directory)
    expected = {int(a.traits['agent_idx']) for a in agents}
    if not result.empty:
        ids = pd.to_numeric(result['agent.agent_idx'],errors='raise').astype(int)
        if ids.duplicated().any() or not set(ids).issubset(expected): raise ValueError('Unexpected response identities.')
        identity = {int(a.traits['agent_idx']):str(a.traits['hhid']) for a in agents}
        for row in result.to_dict('records'):
            if str(row.get('agent.hhid','')) != identity[int(float(row['agent.agent_idx']))]:
                raise ValueError('Response hhid differs from submitted persona.')
    return result


def read_published(path,categories):
    raw = pd.read_csv(path)
    result = {}
    for e in categories:
        config = CONFIGS[e]; size = config['cutoff']//config['width']+1
        for group in ['subsidized','unsubsidized']:
            f=raw.loc[(raw.category==e)&(raw.group==group)].sort_values('bin_index')
            if len(f)!=size or not np.array_equal(f.bin_index,np.arange(size)) or not np.array_equal(f.bin_lower_yuan,np.arange(size)*config['width']):
                raise ValueError('Published histogram bins differ from Figure 1.')
            v=f.fraction.to_numpy(float)
            if not np.isfinite(v).all() or (v<0).any() or abs(v.sum()-1)>.002: raise ValueError('Invalid published fractions.')
            result[e,group]=v
    return result


def analyze(data,run,categories,xing=None,plots=True):
    histograms,tests,pairs,transactions,simulations=[],[],[],[],{}
    expected=set(data.agent_idx.astype(int)); weights=data.set_index('agent_idx').weight_hh
    hashes={}; audits={}
    for e in categories:
        directory=run/e; answers={}
        for arm in ARMS:
            path=directory/f'answers_{arm}_{safe_file_component(MODEL_NAME)}.csv'
            f=pd.read_csv(path,keep_default_na=False)
            if f.agent_idx.duplicated().any() or set(f.agent_idx)!=expected: raise ValueError('All household pairs must be complete: '+e+'/'+arm)
            f=f.sort_values('agent_idx').reset_index(drop=True)
            for row in f.to_dict('records'):
                bill=parse_bill(row['bill_before_discount_yuan'])
                if bill is None: raise ValueError('Invalid saved bill.')
                if arm=='treatment' and parse_complete_answer({bill_question_name(e):bill,USE_QUESTION_NAME:row['coupon_choice']},arm,e) is None:
                    raise ValueError('Invalid coupon choice or eligibility.')
            answers[arm]=f; hashes[str(path)]=file_hash(path)
        matched=match_persona_answers(answers)
        matched=matched.merge(data[['agent_idx','hhid','weight_hh']],on='agent_idx',validate='one_to_one')
        matched['category']=e; pairs.append(matched)
        records=add_outcomes(answers,e).merge(data[['agent_idx','hhid','weight_hh']],on='agent_idx',validate='many_to_one')
        transactions.append(records)
        w=matched.weight_hh.to_numpy(float); config=CONFIGS[e]
        dist={}; y={a:matched[a+'_bill_before_discount_yuan'].to_numpy(float) for a in ARMS}
        for arm in ARMS:
            indices=np.minimum(np.floor(y[arm]/config['width']),config['cutoff']//config['width']).astype(int)
            fractions=np.bincount(indices,weights=w,minlength=config['cutoff']//config['width']+1)/w.sum()
            dist[arm]=fractions
            for i,fraction in enumerate(fractions):
                histograms.append(dict(category=e,arm=arm,bin_index=i,bin_lower_yuan=i*config['width'],
                    bin_upper_yuan=None if i==len(fractions)-1 else (i+1)*config['width'],
                    overflow_bin=int(i==len(fractions)-1),fraction=fraction))
        simulations[e]=dist
        for lower in config['thresholds']:
            upper=lower+config['width']
            c=((y['control']>=lower)&(y['control']<upper)).astype(float)
            t=((y['treatment']>=lower)&(y['treatment']<upper)).astype(float)
            tests.append(dict(category=e,bin_lower_inclusive_yuan=lower,bin_upper_exclusive_yuan=upper,
                weighted_control_bin_share=np.average(c,weights=w),weighted_treatment_bin_share=np.average(t,weights=w),
                **weighted_inference(100*(t-c),w)))
    published=read_published(xing,categories) if xing else None
    out=run/'analysis'/datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f'); out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('weighted_histograms',pd.DataFrame(histograms)),('threshold_bin_inference',pd.DataFrame(tests)),
                       ('matched_persona_pairs',pd.concat(pairs)),('transactions',pd.concat(transactions))]:
        atomic_csv(frame,out/(name+'.csv'))
    if plots: plot_thresholds(simulations,published,categories,out,len(data))
    atomic_json({'answer_hashes':hashes,'n_household_pairs_per_category':len(data),'sample':'all treatment and control personas, including nonredeemers',
        'inference':'paired HC1, t(n-1), indicator differences in percentage points',
        'xing_csv_sha256':file_hash(xing) if xing else None,'xing_note':'Published recovered histogram fractions; human groups are redemption-status groups, not receipt arms.',
        'model_calls':0},out/'analysis_audit.json')
    return pd.DataFrame(tests),out


def plot_thresholds(simulations,published,categories,out,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols=4 if published else 2
    fig,axes=plt.subplots(len(categories),cols,figsize=(3*cols,2.7*len(categories)),squeeze=False)
    for row,e in enumerate(categories):
        cfg=CONFIGS[e]
        series=[simulations[e]['treatment'],simulations[e]['control']]
        titles=['LLM: coupon receipt','LLM: no receipt']
        if published:
            series += [published[e,'subsidized'],published[e,'unsubsidized']]
            titles += ['Xing: subsidized','Xing: unsubsidized']
        ceiling=min(1.,max(.4,np.ceil((max(v.max() for v in series)+.025)/.2)*.2))
        x=np.arange(len(series[0]))*cfg['width']
        for col,values in enumerate(series):
            ax=axes[row,col]; ax.bar(x,values,width=cfg['width']*.9,align='edge',color='#7EAF99' if col%2==0 else '#AFB1B4')
            for threshold in cfg['thresholds']: ax.axvline(threshold+cfg['width']/2,color='grey',ls='--',lw=.8)
            ax.set_ylim(0,ceiling); ax.set_xlim(0,cfg['cutoff']+cfg['width'])
            ax.set_xticks(cfg['ticks'],[f'{v:,}+' if v==cfg['cutoff'] else f'{v:,}–{v+cfg["width"]:,}' for v in cfg['ticks']],rotation=45,ha='right',fontsize=7)
            ax.set_title(titles[col]+(f' (n={n})' if col<2 else ''),fontsize=9)
            if col==0: ax.set_ylabel(cfg['name']+'\nFraction')
    fig.tight_layout(); fig.savefig(out/'threshold_distributions.png',dpi=200); fig.savefig(out/'threshold_distributions.pdf'); plt.close(fig)


def cli(argv=None):
    global BATCH_SIZE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','collect','analyze'],default='prepare')
    parser.add_argument('--input',type=Path,default=SCRIPT_DIR/'outputs/step1/personas.csv')
    parser.add_argument('--output-dir',type=Path,default=SCRIPT_DIR/'outputs/step3')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--categories',nargs='+',choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument('--batch-size',type=int,default=100)
    parser.add_argument('--expected-count',type=int)
    parser.add_argument('--xing-csv',type=Path,help='Optional existing published histogram CSV for Figure 1 comparison.')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    if args.batch_size<1: parser.error('Batch size must be positive.')
    if args.categories and len(set(args.categories))!=len(args.categories): parser.error('Duplicate categories.')
    if args.run_dir and args.categories: parser.error('An existing run uses the saved categories.')
    if args.mode=='analyze' and not args.run_dir: parser.error('--mode analyze requires --run-dir.')
    load_edsl()
    if args.run_dir is None:
        data=load_personas(args.input,args.expected_count)
        args.run_dir=prepare(data,args.input,args.output_dir/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f'),
                             args.categories or list(EXPERIMENT_CONFIGS),args.batch_size)
    import fcntl
    with (args.run_dir/'.step3.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data,agents,spec=load_run(args.run_dir)
        if args.expected_count is not None and len(data)!=args.expected_count: raise ValueError('Unexpected sample size.')
        BATCH_SIZE=spec['batch_size']
        if args.mode=='prepare': print('Offline Step 3 ready:',args.run_dir); return
        if args.mode=='collect':
            marker = args.run_dir/'collection_started.json'
            if marker.exists():
                raise RuntimeError('Collection has already started for this run; it cannot be submitted again.')
            atomic_json({'collection': 'single pass'}, marker)
            from edsl import Model
            model=Model(MODEL_NAME,service_name=MODEL_SERVICE,**MODEL_PARAMETERS)
            for e in spec['categories']:
                for arm in ARMS:
                    run_arm_batches(arm,make_survey(e,arm),make_scenario(e,arm),model,agents,e,str(args.run_dir/e))
                    answers=combine_arm_answers(arm,MODEL_NAME,e,str(args.run_dir/e))
                    if set(answers.agent_idx)!=set(data.agent_idx): raise ValueError('Collection incomplete; no analysis is produced.')
        _,out=analyze(data,args.run_dir,spec['categories'],args.xing_csv,not args.no_plots)
        print('Offline analysis:',out)


if __name__ == '__main__':
    cli()
