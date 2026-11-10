"""Step 1: extract personas from existing private CHFS inputs.

No CHFS observations, credentials, local project paths, or historical imports
are embedded. Input column names are a documented, configurable contract.
Run --self-test to exercise the complete pipeline with artificial households.
"""

APPLICATION = 'digital_coupon'

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SURVEY_YEAR = 2011 if APPLICATION == "home_appliances" else 2019
POLICY_YEAR = 2008 if APPLICATION == "home_appliances" else 2020
WEIGHT_NAME = "swgt" if APPLICATION == "home_appliances" else "weight_hh"
DURABLES = ["摄像机/照相机", "黑白/彩色电视机", "洗衣机", "冰箱", "空调",
            "电子计算机/电脑", "音响", "太阳能/电能热水器", "家具", "卫星接收器", "乐器"]
REGIONS = {
    "Eastern": "北京市 天津市 河北省 上海市 江苏省 浙江省 福建省 山东省 广东省 海南省".split(),
    "Central": "山西省 安徽省 江西省 河南省 湖北省 湖南省".split(),
    "Northeastern": "辽宁省 吉林省 黑龙江省".split(),
    "Western": "内蒙古自治区 广西壮族自治区 重庆市 四川省 贵州省 云南省 西藏自治区 陕西省 甘肃省 青海省 宁夏回族自治区 新疆维吾尔自治区".split(),
}
PROVINCE_REGION = {p: region for region, provinces in REGIONS.items() for p in provinces}
def default_config():
    """Point to existing private files; no intermediate tables are required."""
    appliance = APPLICATION == "home_appliances"
    hh = dict(hhid="hhid", province="prov_CHN", area_type="area_type",
              household_size="total_population", household_income="hh_income",
              gender="a2003", birth_year="a2005", children_under_14="children_under_14",
              seniors_60_or_over="seniors_60_or_over", weight=WEIGHT_NAME)
    if appliance:
        hh["durable_spending"] = "g1014_imp"
        hh.update({f"owns_{i}": f"c8001_{i}_mc" for i in range(1, 12)})
    else:
        hh.update(city_level="city_level", city_label="city_lab",
                  annual_consumption="total_consump", food_spending="g1001_imp",
                  entertainment_spending="g1010_imp", tele_spending="g1009g_imp",
                  travel_spending="g1018_imp")
    return {
        "schema_version": 2, "application": APPLICATION,
        "input_mapping_confirmed": False,
        "files": {"households": None, "members": None,
                  "master": "农村户主样本（4）.csv" if appliance else "chfs2019_master_202112.dta"},
        "columns": {
            "households": hh,
            "members": dict(hhid="hhid", member_id="pid", is_head="hhead", gender="a2003", birth_year="a2005"),
            "master": dict(hhid="hhid", weight=WEIGHT_NAME),
        },
        "weight_fallback_table": "master",
        "head_value": "1", "household_merges": [], "household_source_constants": {},
        "area_values": {"city": ["city"], "rural": ["rural"]},
        "missing_values": {}, "composition_year": SURVEY_YEAR,
        "elderly_age_at_least": 60, "children_age_strictly_less_than": 14,
        "missing_core_policy": "error" if appliance else "exclude",
        "wealth_inputs": {"table": "households", "total_assets_column": "total_assets",
                          "total_liabilities_column": "total_liabilities", "net_worth_column": "net_worth",
                          "unit_to_rmb": 1, "identity_tolerance_rmb": 0.01},
    }



def number(series, label):
    out = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & (out.isna() | ~np.isfinite(out))
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} nonnumeric/nonfinite values; map release missing codes explicitly.")
    return out.astype(float)


def identifiers(series, label):
    out = series.astype("string").str.strip().str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
    if out.isna().any() or out.eq("").any():
        raise ValueError(f"{label}: missing identifiers.")
    return out


def code_tokens(series):
    """Make numeric Stata codes and equivalent CSV strings agree."""
    return series.astype("string").str.strip().str.replace(r"^(-?[0-9]+)\.0+$", r"\1", regex=True)


def read_table(path):
    if not path.is_file():
        raise FileNotFoundError(f"Private input not found: {path}. Supply your licensed CHFS table; no fallback data are used.")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    raise ValueError("Inputs must be .csv or .dta files.")


def load_tables(input_dir, config):
    """Read extra existing sources only for fields absent from household inputs."""
    tables, hashes, raw_cache = {}, {}, {}

    def load_source(name):
        if name in tables:
            return tables[name]
        filenames = config["files"].get(name)
        if not filenames:
            raise ValueError(f"Needed source {name!r} is not specified. Map fields already in your household file or point to an existing source; do not create a new intermediate table.")
        if not isinstance(filenames, list):
            filenames = [filenames]
        if name != "households" and len(filenames) != 1:
            raise ValueError("Only the primary household input supports multiple source files.")
        frames = []
        for filename in filenames:
            path = Path(filename)
            if not path.is_absolute():
                path = input_dir / path
            cache_key = str(path.resolve())
            if cache_key not in raw_cache:
                raw_cache[cache_key] = read_table(path)
            frame = raw_cache[cache_key].copy()
            if frame.columns.duplicated().any():
                raise ValueError(f"{name}: duplicate column names.")
            for column, codes in config["missing_values"].get(name, {}).items():
                if column not in frame:
                    raise ValueError(f"Unknown missing-value mapping: {name}.{column}")
                frame.loc[code_tokens(frame[column]).isin({str(x) for x in codes}), column] = pd.NA
            if name == "households":
                constants = config.get("household_source_constants", {})
                for column, value in constants.get(filename, constants.get(path.name, {})).items():
                    if column not in frame:
                        frame[column] = value  # Supplied columns always take precedence.
            id_column = config["columns"][name]["hhid"]
            if id_column not in frame:
                raise ValueError(f"{name}: missing household identifier column {id_column}.")
            frame["_hhid"] = identifiers(frame[id_column], f"{name}.hhid")
            frames.append(frame)
            hashes[f"{name}:{filename}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[name] = pd.concat(frames, ignore_index=True)
        return tables[name]

    load_source("households")
    for merge in config.get("household_merges", []):
        missing_columns = {dest: source for dest, source in merge["columns"].items()
                           if dest not in tables["households"].columns}
        if not missing_columns:
            continue  # All requested data already exist: do not read another file.
        source_name = merge["table"]
        if source_name == "households":
            raise ValueError("A fallback join must name another existing source.")
        source = load_source(source_name)
        if not set(missing_columns.values()).issubset(source.columns):
            raise ValueError("A configured source column is absent from the existing master table.")
        if len(set(missing_columns.values())) != len(missing_columns):
            raise ValueError("Fallback source columns must map uniquely.")
        selected = source[["_hhid"] + list(missing_columns.values())]
        if selected.groupby("_hhid").nunique(dropna=False).gt(1).any().any():
            raise ValueError("Conflicting repeated master-table values.")
        selected = selected.drop_duplicates("_hhid").rename(columns={v: k for k, v in missing_columns.items()})
        tables["households"] = tables["households"].merge(selected, on="_hhid", how="left", sort=False, validate="many_to_one")
    hh_columns = config["columns"]["households"]
    demographics = ["gender", "birth_year", "children_under_14", "seniors_60_or_over"]
    if any(hh_columns[field] not in tables["households"] for field in demographics):
        load_source("members")
    if hh_columns["weight"] not in tables["households"]:
        load_source(config["weight_fallback_table"])
    wealth_table = config["wealth_inputs"]["table"]
    if wealth_table != "households":
        load_source(wealth_table)
    return tables, hashes



def mapped(frame, config, table, field):
    return frame[config["columns"][table][field]]


def scalar_by_household(frame, column, ids, label):
    if column not in frame:
        raise ValueError(f"{label}: missing amount column {column}.")
    selected = frame.loc[frame._hhid.isin(ids), ["_hhid", column]].copy()
    selected[column] = number(selected[column], label)
    if selected.groupby("_hhid")[column].nunique(dropna=False).gt(1).any():
        raise ValueError(f"{label}: conflicting repeated household values.")
    return selected.drop_duplicates("_hhid").set_index("_hhid")[column].reindex(ids)


def read_wealth_inputs(tables, ids, config):
    """Read privately constructed wealth measures; do not build components."""
    spec = config["wealth_inputs"]
    scale = float(spec["unit_to_rmb"])
    tolerance = float(spec["identity_tolerance_rmb"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Wealth unit_to_rmb must be positive.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Wealth identity tolerance must be finite and nonnegative.")
    frame = tables[spec["table"]]
    values = pd.DataFrame(index=ids)
    for name in ["total_assets", "total_liabilities", "net_worth"]:
        values[name] = scalar_by_household(frame, spec[name + "_column"], ids, name) * scale
    complete = values.notna().all(axis=1)
    consistent = np.isclose(values.loc[complete, "total_assets"] - values.loc[complete, "total_liabilities"],
                            values.loc[complete, "net_worth"], rtol=0, atol=tolerance)
    if not consistent.all():
        raise ValueError("Provided assets, liabilities and net worth do not satisfy the accounting identity within the configured tolerance; values were not replaced.")
    values["wealth_missing_value_count"] = values.isna().sum(axis=1)
    return values


def demographics_from_existing_inputs(tables, indexed, ids, config):
    """Use supplied fields first; derive only absent fields from an existing roster."""
    mapping = config["columns"]["households"]
    fields = {"gender_code": "gender", "birth_year": "birth_year",
              "children_under_14": "children_under_14", "seniors_60_or_over": "seniors_60_or_over"}
    result = pd.DataFrame(index=ids)
    missing = []
    sources = {}
    for output, field in fields.items():
        column = mapping[field]
        if column in indexed:
            result[output] = number(indexed[column], column).reindex(ids)
            sources[output] = "existing_household_field"
        else:
            missing.append(output)
    result["roster_birth_year_missing_count"] = np.nan
    result["household_size_roster"] = np.nan
    if missing:
        if "members" not in tables:
            raise ValueError("Some demographic fields are absent; map an existing household column or supply the existing roster file.")
        members = tables["members"].loc[tables["members"]._hhid.isin(ids)].copy()
        members["_member_id"] = identifiers(mapped(members, config, "members", "member_id"), "member_id")
        if members.duplicated(["_hhid", "_member_id"]).any():
            raise ValueError("Duplicate household/member identifier.")
        birth = number(mapped(members, config, "members", "birth_year"), "member birth year")
        invalid = birth.notna() & ((birth % 1 != 0) | (birth > SURVEY_YEAR) | (birth < SURVEY_YEAR - 120))
        if invalid.any():
            raise ValueError("Invalid member birth years; review source missing-value codes.")
        members["_birth_year"] = birth
        roster = members.groupby("_hhid")
        size = roster.size().reindex(ids)
        if size.isna().any():
            raise ValueError("Some selected households are absent from the existing roster.")
        unknown = roster._birth_year.apply(lambda x: x.isna().sum()).reindex(ids)
        result["roster_birth_year_missing_count"] = unknown
        result["household_size_roster"] = size
        if any(name in missing for name in ["gender_code", "birth_year"]):
            is_head = code_tokens(mapped(members, config, "members", "is_head")).eq(str(config["head_value"]))
            counts = members.assign(_head=is_head).groupby("_hhid")._head.sum().reindex(ids, fill_value=0)
            if not counts.eq(1).all():
                raise ValueError("Each selected household needs exactly one identified head.")
            head = members.loc[is_head].set_index("_hhid")
            if "gender_code" in missing:
                result["gender_code"] = number(mapped(head, config, "members", "gender"), "head gender").reindex(ids)
            if "birth_year" in missing:
                result["birth_year"] = head._birth_year.reindex(ids)
        age = config["composition_year"] - birth
        for name, values in [("children_under_14", age.ge(0) & age.lt(config["children_age_strictly_less_than"])),
                             ("seniors_60_or_over", age.ge(config["elderly_age_at_least"]))]:
            if name in missing:
                counts = values.groupby(members._hhid).sum().reindex(ids)
                result[name] = counts.where(unknown.eq(0))
        sources.update({name: "existing_member_source" for name in missing})
    for field in ["children_under_14", "seniors_60_or_over"]:
        values = result[field].dropna()
        if (values.lt(0) | (values % 1 != 0)).any():
            raise ValueError("Household member counts must be nonnegative integers.")
    if not result.gender_code.dropna().isin([1, 2]).all():
        raise ValueError("Head gender must be coded 1/2; map missing codes explicitly.")
    return result, sources



def construct_personas(tables, config, sample_ids=None, expected_count=None):
    if config.get("application") != APPLICATION or config.get("schema_version") != 2:
        raise ValueError("Configuration belongs to a different application or schema version.")
    if not config["input_mapping_confirmed"]:
        raise ValueError("The supplied template contains assumed column names. Review your licensed release and set input_mapping_confirmed=true.")
    hh = tables["households"].copy()
    if hh._hhid.duplicated().any():
        raise ValueError("Household table must contain exactly one row per hhid.")
    province = mapped(hh, config, "households", "province").astype("string").str.strip()
    area_source = code_tokens(mapped(hh, config, "households", "area_type"))
    area_map = {}
    for area, values in config["area_values"].items():
        if area not in {"city", "rural"}:
            raise ValueError("area_values keys must be city and rural.")
        for value in values:
            if str(value) in area_map:
                raise ValueError("Overlapping urban/rural source codes.")
            area_map[str(value)] = area
    area = area_source.map(area_map)
    if area.isna().any():
        raise ValueError("Unknown/missing area codes; configure area_values without guessing rural status.")
    eligible = area.eq("rural") if APPLICATION == "home_appliances" else (
        province.eq("浙江省") & number(mapped(hh, config, "households", "city_level"), "city_level").eq(2))
    selected = hh.loc[eligible].copy()
    if sample_ids is not None:
        requested = pd.Index(sample_ids, name="hhid")
        if requested.has_duplicates or len(requested.difference(selected._hhid)):
            raise ValueError("Private sample-ID list contains duplicates, absent IDs, or ineligible households.")
        selected = selected.set_index("_hhid", drop=False).loc[requested].reset_index(drop=True)
    elif APPLICATION == "digital_coupon":
        selected["_area_order"] = area.loc[selected.index].map({"city": 0, "rural": 1})
        selected = selected.sort_values("_area_order", kind="stable")
    if selected.empty:
        raise ValueError("No households satisfy the application selection.")
    ids = pd.Index(selected._hhid, name="hhid")
    profile = pd.DataFrame(index=ids)
    indexed = hh.set_index("_hhid")
    profile["province"] = pd.Series(province.to_numpy(), index=hh._hhid).reindex(ids)
    profile["area_type"] = pd.Series(area.to_numpy(), index=hh._hhid).reindex(ids)
    profile["region"] = profile.province.map(PROVINCE_REGION)
    if profile.region.isna().any():
        raise ValueError("Unrecognized province name; reconcile it with the documented region mapping.")
    for output, source in [("total_population", "household_size"), ("hh_income", "household_income")]:
        profile[output] = number(mapped(indexed, config, "households", source), source).reindex(ids)
    demographics, demographic_sources = demographics_from_existing_inputs(tables, indexed, ids, config)
    profile = profile.join(demographics)
    profile["household_size_conflict"] = profile.total_population.ne(profile.household_size_roster).where(
        profile.household_size_roster.notna())
    profile["head_age_survey_year"] = SURVEY_YEAR - profile.birth_year
    # Same historical prompt convention: birth years <=1900 are rendered unknown.
    profile["head_age_policy_year"] = (POLICY_YEAR - profile.birth_year).where(
        profile.birth_year.gt(1900) & profile.birth_year.le(POLICY_YEAR))
    wealth = read_wealth_inputs(tables, ids, config)
    profile = profile.join(wealth)
    existing_weight = config["columns"]["households"]["weight"]
    weight_source = "households" if existing_weight in hh.columns else config["weight_fallback_table"]
    weight_col = existing_weight if weight_source == "households" else config["columns"][weight_source]["weight"]
    profile[WEIGHT_NAME] = scalar_by_household(tables[weight_source], weight_col, ids, "sampling weight")
    core = ["gender_code", "birth_year", "total_population", "hh_income", "total_assets", "total_liabilities",
            "net_worth", "children_under_14", "seniors_60_or_over"]
    if APPLICATION == "digital_coupon":
        profile["city_level"] = number(mapped(indexed, config, "households", "city_level"), "city level").reindex(ids)
        profile["city_label"] = mapped(indexed, config, "households", "city_label").astype("string").reindex(ids)
        core.append("city_label")
        for field in ["annual_consumption", "food_spending", "entertainment_spending", "tele_spending", "travel_spending"]:
            profile[field] = number(mapped(indexed, config, "households", field), field).reindex(ids)
            core.append(field)
        # Travel remains part of the original complete-case source selection,
        # even though the submitted full-persona questionnaire omits that sentence.
    else:
        profile["last_year_durable_spending"] = number(mapped(indexed, config, "households", "durable_spending"), "durable spending").reindex(ids)
        for i in range(1, 12):
            values = number(mapped(indexed, config, "households", f"owns_{i}"), f"owns_{i}").reindex(ids)
            if not values.isin([0, 1]).all():
                raise ValueError("Durable ownership must be observed 0/1; missing ownership is not nonownership.")
            profile[f"c8001_{i}_mc"] = values.astype(int)
        profile["owned_durable_text"] = profile.apply(lambda r: "、".join(label for i, label in enumerate(DURABLES, 1) if r[f"c8001_{i}_mc"] == 1) or "无", axis=1)
        profile["not_owned_durable_text"] = profile.apply(lambda r: "、".join(label for i, label in enumerate(DURABLES, 1) if r[f"c8001_{i}_mc"] == 0) or "无", axis=1)
        profile["durable_spending_missing"] = profile.last_year_durable_spending.isna()
    if (profile.total_population.dropna().le(0) | (profile.total_population.dropna() % 1 != 0)).any():
        raise ValueError("Reported household size must be a positive integer.")
    missing_core = profile[core].isna().any(axis=1)
    if config["missing_core_policy"] not in {"error", "exclude"}:
        raise ValueError("missing_core_policy must be error or exclude.")
    if missing_core.any() and config["missing_core_policy"] == "error":
        raise ValueError("Missing required persona fields; no silent zero filling or household removal.")
    if sample_ids is not None and missing_core.any():
        raise ValueError("Requested frozen sample contains incomplete households; sample membership was not changed.")
    profile = profile.loc[~missing_core].copy()
    if profile[WEIGHT_NAME].isna().any() or profile[WEIGHT_NAME].le(0).any():
        raise ValueError("Every retained household needs a matched positive sampling weight; no unit-weight fallback.")
    if profile.empty:
        raise ValueError("No usable personas remain.")
    if expected_count is not None and len(profile) != expected_count:
        raise ValueError(f"Expected {expected_count} households but constructed {len(profile)}; no truncation or padding.")
    profile.insert(0, "agent_idx", range(len(profile)))
    audit = dict(application=APPLICATION, survey_year=SURVEY_YEAR, policy_year=POLICY_YEAR,
                 composition_year=config["composition_year"], households_input=len(hh),
                 households_eligible=len(ids), personas_output=len(profile), complete_case_rows_removed=int(missing_core.sum()),
                 sampling_weight=WEIGHT_NAME, weight_source=weight_source, demographic_sources=demographic_sources, weights_rescaled=False,
                 household_size_conflicts=int(profile.household_size_conflict.sum()),
                 unknown_policy_age=int(profile.head_age_policy_year.isna().sum()),
                 wealth_missing_households=int(profile.wealth_missing_value_count.gt(0).sum()),
                 durable_spending_missing=int(profile.durable_spending_missing.sum()) if APPLICATION == "home_appliances" else None)
    return profile.reset_index(), audit


def synthetic_tables():
    """One invented existing household file already contains all needed fields."""
    cfg = default_config()
    cfg["input_mapping_confirmed"] = True
    cfg["files"]["households"] = "synthetic_existing_input.csv"
    hh = pd.DataFrame({"hhid": ["SYNTH_001", "SYNTH_002"], "prov_CHN": ["浙江省", "浙江省"],
                       "area_type": ["rural", "rural"], "total_population": [4, 2], "hh_income": [12345, -123],
                       "a2003": [1, 2], "a2005": [SURVEY_YEAR-40, SURVEY_YEAR-30],
                       "children_under_14": [1, 0], "seniors_60_or_over": [2, 0], WEIGHT_NAME: [2.5, 5.0],
                       "city_level": [2, 2], "city_lab": ["Synthetic city", "Synthetic city"],
                       "total_assets": [1500, 2500], "total_liabilities": [300, 400], "net_worth": [1200, 2100],
                       "total_consump": [3000, 4000], "g1001_imp": [100, 200], "g1010_imp": [10, 20],
                       "g1009g_imp": [120, 240], "g1018_imp": [0, 1], "g1014_imp": [50, np.nan]})
    for i in range(1, 12):
        hh[f"c8001_{i}_mc"] = [i % 2, (i + 1) % 2]
    hh["_hhid"] = hh.hhid.astype("string")
    return {"households": hh}, cfg



def synthetic_members():
    frame = pd.DataFrame({"hhid": ["SYNTH_001"]*4 + ["SYNTH_002"]*2,
                          "pid": ["p1", "p2", "p3", "p4", "p1", "p2"],
                          "hhead": ["1", "0", "0", "0", "1", "0"], "a2003": [1, 2, 1, 2, 2, 1],
                          "a2005": [SURVEY_YEAR-40, SURVEY_YEAR-60, SURVEY_YEAR-61, SURVEY_YEAR-13,
                                    SURVEY_YEAR-30, SURVEY_YEAR-14]})
    frame["_hhid"] = frame.hhid.astype("string")
    return frame



def self_test():
    tables, cfg = synthetic_tables()
    frame, audit = construct_personas(tables, cfg, expected_count=2)
    assert frame.seniors_60_or_over.tolist() == [2, 0]
    assert frame.children_under_14.tolist() == [1, 0]
    assert frame[WEIGHT_NAME].tolist() == [2.5, 5.0]
    assert frame.hh_income.iloc[1] == -123
    assert audit["weight_source"] == "households"
    assert set(audit["demographic_sources"].values()) == {"existing_household_field"}
    def rejects(changed, configuration, **kwargs):
        try:
            construct_personas(changed, configuration, **kwargs)
        except (ValueError, KeyError):
            return
        raise AssertionError("Invalid synthetic input was accepted.")
    changed = deepcopy(tables); changed["households"].loc[0, WEIGHT_NAME] = 0
    rejects(changed, cfg)
    changed = deepcopy(tables); changed["households"].loc[0, "net_worth"] = 999
    rejects(changed, cfg)
    changed = deepcopy(tables); changed["households"]["net_worth"] = changed["households"].net_worth.astype(float)
    changed["households"].loc[0, "net_worth"] = 1200.005
    supplied, _ = construct_personas(changed, cfg)
    assert supplied.net_worth.iloc[0] == 1200.005
    rejects(tables, cfg, expected_count=547)
    reverse, _ = construct_personas(tables, cfg, sample_ids=["SYNTH_002", "SYNTH_001"])
    assert reverse.hhid.tolist() == ["SYNTH_002", "SYNTH_001"]
    # Existing roster is used only for fields absent from the household table.
    roster_tables = deepcopy(tables)
    roster_tables["households"] = roster_tables["households"].drop(columns=["a2003", "a2005", "children_under_14", "seniors_60_or_over"])
    roster_tables["members"] = synthetic_members()
    roster_frame, roster_audit = construct_personas(roster_tables, cfg)
    assert roster_frame.seniors_60_or_over.tolist() == [2, 0]
    assert roster_frame.children_under_14.tolist() == [1, 0]
    partial = deepcopy(tables)
    partial["households"] = partial["households"].drop(columns="seniors_60_or_over")
    partial["members"] = synthetic_members()
    partial_frame, partial_audit = construct_personas(partial, cfg)
    assert partial_audit["demographic_sources"]["children_under_14"] == "existing_household_field"
    assert partial_audit["demographic_sources"]["seniors_60_or_over"] == "existing_member_source"
    assert partial_frame.seniors_60_or_over.tolist() == [2, 0]
    younger = deepcopy(roster_tables); younger["members"].loc[1, "a2005"] = SURVEY_YEAR-59
    younger_frame, _ = construct_personas(younger, cfg)
    assert younger_frame.seniors_60_or_over.tolist() == [1, 0]
    duplicate = deepcopy(roster_tables)
    duplicate["members"] = pd.concat([duplicate["members"], duplicate["members"].iloc[[0]]])
    rejects(duplicate, cfg)
    # Existing master supplies weights only when absent from the primary input.
    fallback = deepcopy(tables)
    fallback["master"] = fallback["households"][["_hhid", "hhid", WEIGHT_NAME]].copy()
    fallback["households"] = fallback["households"].drop(columns=WEIGHT_NAME)
    matched, matched_audit = construct_personas(fallback, cfg)
    assert matched[WEIGHT_NAME].tolist() == [2.5, 5.0]
    assert matched_audit["weight_source"] == "master"
    bad = deepcopy(fallback); extra = bad["master"].iloc[[0]].copy(); extra[WEIGHT_NAME] = 99
    bad["master"] = pd.concat([bad["master"], extra]); rejects(bad, cfg)
    missing = deepcopy(fallback); missing["master"] = missing["master"].iloc[1:]; rejects(missing, cfg)
    if APPLICATION == "home_appliances":
        assert frame.durable_spending_missing.tolist() == [False, True]
        bad = deepcopy(tables); bad["households"].loc[0, "c8001_1_mc"] = np.nan
        rejects(bad, cfg)
    else:
        missing = deepcopy(tables); missing["households"].loc[0, "g1018_imp"] = np.nan
        result, info = construct_personas(missing, cfg)
        assert len(result) == 1 and info["complete_case_rows_removed"] == 1
    with tempfile.TemporaryDirectory(prefix="synthetic_existing_inputs_") as directory:
        root = Path(directory)
        tables["households"].drop(columns="_hhid").to_csv(root/cfg["files"]["households"], index=False)
        # No member/master file exists. Neither is opened when all fields exist.
        config = deepcopy(cfg); config["files"]["members"] = "absent_roster.csv"
        config["files"]["master"] = "absent_master.dta"
        loaded, hashes = load_tables(root, config)
        assert set(loaded) == {"households"} and len(hashes) == 1
        reread, _ = construct_personas(loaded, config)
        pd.testing.assert_frame_equal(frame, reread)
        # Redundant configured joins also do not cause another file read.
        config["household_merges"] = [{"table": "master", "columns": {"hh_income": "hh_income"}}]
        loaded, hashes = load_tables(root, config)
        assert set(loaded) == {"households"} and len(hashes) == 1
        # A genuinely missing weight is matched from an existing master fixture.
        master_config = deepcopy(cfg)
        master_config["files"]["master"] = "synthetic_existing_master.csv"
        fallback["households"].drop(columns="_hhid").to_csv(root/cfg["files"]["households"], index=False)
        fallback["master"].drop(columns="_hhid").to_csv(root/master_config["files"]["master"], index=False)
        loaded, hashes = load_tables(root, master_config)
        result, _ = construct_personas(loaded, master_config)
        assert set(loaded) == {"households", "master"} and len(hashes) == 2
        assert result[WEIGHT_NAME].tolist() == [2.5, 5.0]
        # Stata input works without intermediate CSV exports.
        dta = deepcopy(cfg); dta["files"]["households"] = "synthetic_existing_input.dta"
        tables["households"].drop(columns="_hhid").to_stata(root/dta["files"]["households"], write_index=False, version=118)
        loaded, hashes = load_tables(root, dta)
        result, _ = construct_personas(loaded, dta)
        pd.testing.assert_frame_equal(frame, result)
    print(f"PASS: {APPLICATION}; no redundant inputs, inclusive age 60, synthetic records only.")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("step1_input_template.json"))
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "data/private")
    parser.add_argument("--input", type=Path, nargs="+", help="Existing primary household file(s); no intermediate exports required.")
    parser.add_argument("--member-source", type=Path, help="Existing roster, used only for demographic columns absent from the primary input.")
    parser.add_argument("--master-source", type=Path, help="Existing master/rural table, used only for absent weights or configured fields.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/step1")
    parser.add_argument("--sample-ids", type=Path, help="Optional private CSV with hhid in the required survey order.")
    parser.add_argument("--expected-count", type=int, help="Fail, never truncate/pad, if the resulting sample has another size.")
    parser.add_argument("--init-config", type=Path, help="Write an assumed input mapping without reading any data.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.init_config:
        with args.init_config.open("x", encoding="utf-8") as handle:
            json.dump(default_config(), handle, ensure_ascii=False, indent=2); handle.write("\n")
        print(f"Created input template: {args.init_config}"); return
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive.")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    # Check the mapping before attempting to open any private table.
    if not cfg.get("input_mapping_confirmed"):
        parser.error("Review step1_input_template.json against your licensed release and confirm the input mapping first.")
    if args.input:
        cfg["files"]["households"] = [str(path.resolve()) for path in args.input]
    if args.member_source:
        cfg["files"]["members"] = str(args.member_source.resolve())
    if args.master_source:
        cfg["files"]["master"] = str(args.master_source.resolve())
    if not cfg["files"].get("households"):
        parser.error("Specify your existing household input using --input or files.households; no new input table needs to be created.")
    tables, hashes = load_tables(args.input_dir, cfg)
    ids = None
    if args.sample_ids:
        sample = read_table(args.sample_ids)
        if "hhid" not in sample: parser.error("--sample-ids must contain hhid.")
        ids = identifiers(sample.hhid, "sample_ids").tolist()
    personas, audit = construct_personas(tables, cfg, ids, args.expected_count)
    audit.update(input_sha256=hashes, configuration=cfg, synthetic_data=False,
                 sample_ids_sha256=hashlib.sha256(args.sample_ids.read_bytes()).hexdigest() if args.sample_ids else None)
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing persona outputs are never overwritten.")
    args.output_dir.mkdir(parents=True)
    personas.to_csv(args.output_dir / "personas.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "step1_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(personas)} personas in {args.output_dir}; outputs contain private household data and stay local.")


if __name__ == "__main__":
    main()
