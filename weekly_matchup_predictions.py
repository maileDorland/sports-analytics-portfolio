from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import math

# Matchups / Home field
matchup = [
    ('Cowboys', 'Commanders'), ('Lions', 'Vikings'), ('Broncos', 'Chiefs'),
    ('Texans', 'Chargers'), ('Ravens', 'Packers'), ('Seahawks', 'Panthers'),
    ('Cardinals', 'Bengals'), ('Steelers', 'Browns'), ('Jaguars', 'Colts'),
    ('Buccaneers', 'Dolphins'), ('Patriots', 'Jets'), ('Saints', 'Titans'),
    ('Giants', 'Raiders'), ('Eagles', 'Bills'), ('Bears', '49ers'),
    ('Rams', 'Falcons')
]
home_field = [t[1] for t in matchup]

week_15_bye = []
week_16_bye = []

# Power rankings
power_rank_nfl = [
    'Seahawks', 'Patriots', 'Bills', 'Broncos', 'Jaguars', 'Bears', '49ers',
    'Rams', 'Chargers', 'Texans', 'Eagles', 'Packers', 'Steelers', 'Panthers',
    'Lions', 'Ravens', 'Colts', 'Buccaneers', 'Vikings', 'Cowboys',
    'Chiefs', 'Falcons', 'Bengals', 'Saints', 'Dolphins', 'Commanders',
    'Cardinals', 'Titans', 'Browns', 'Jets', 'Giants', 'Raiders'
]

power_rank_espn = [
    'Seahawks', 'Rams', 'Broncos', 'Patriots', 'Jaguars', '49ers', 'Bills',
    'Bears', 'Chargers', 'Eagles', 'Packers', 'Steelers', 'Texans', 'Lions',
    'Colts', 'Panthers', 'Buccaneers', 'Ravens', 'Vikings', 'Cowboys',
    'Falcons', 'Chiefs', 'Dolphins', 'Saints', 'Bengals', 'Commanders',
    'Titans', 'Cardinals', 'Browns', 'Jets', 'Giants', 'Raiders'
]

# Load data
df = pd.read_excel(r"C:\Users\might\OneDrive\Documents\Football_datasets\week12_stats.xlsx")
depth_chart = pd.read_excel(r"C:\Users\might\OneDrive\Documents\Football_datasets\team_depth_chart.xlsx")

# Helpers

def get_team_data(side, team, play):
    sub = df[(df['Sides'] == side) & (df['Team'] == team) & (df['Play'] == play)]
    if sub.empty:
        return {}
    row_dict = sub.iloc[0].to_dict()
    dictionary = {k: v for k, v in row_dict.items() if not pd.isna(v)}
    return dictionary

def safe_rate(numer, denom):
    if numer is None or denom in (None, 0):
        return 0.0
    try:
        return float(numer) / float(denom)
    except ZeroDivisionError:
        return 0.0

def parse_long_gain(lng):
    if lng is None:
        return None
    if isinstance(lng, (int, float)):
        return float(lng)
    s = str(lng)
    digits = ''.join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else None

def power_ranking(team):
    n_teams = len(power_rank_nfl)
    rank_nfl = power_rank_nfl.index(team) + 1
    rank_espn = power_rank_espn.index(team) + 1
    avg_rank = (rank_nfl + rank_espn) / 2.0

    raw = 1 - (avg_rank - 1) / (n_teams - 1)   # 1.0 best, 0.0 worst
    compressed = 0.5 + 0.5 * (raw - 0.5)
    return compressed

# Position-based injury parsing + weighting

NO_PLAY = {"O", "OUT", "IR", "PUP", "SUSP", "DNP"}   # won’t play
MAYBE_PLAY = {"Q", "D", "DOUBT"}                     # questionable/doubtful
YES_PLAY = {"A","PROB"}                        # active/probable (ignored)

def normalize_status(x) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()

def normalize_pos(pos: str) -> str:
    if pos is None:
        return ""
    p = str(pos).strip().upper()

    # offense
    if p in {"QB", "RB", "FB", "TE", "WR", "LT", "LG", "C", "RG", "RT"}:
        return p

    # defensive line
    if p in {"LDE", "RDE", "DE"}:
        return "DE"
    if p in {"LDT", "RDT", "DT"}:
        return "DT"

    # linebackers
    if p in {"WLB", "MLB", "SLB", "LB"}:
        return "LB"

    # corners / nickel
    if p in {"LCB", "RCB", "CB", "NB"}:
        return "CB"

    # safeties
    if p in {"SS", "FS", "S"}:
        return "S"

    # special teams
    if p in {"PK", "K"}:
        return "PK"
    if p == "P":
        return "P"
    if p == "LS":
        return "LS"
    if p == "PR":
        return "PR"
    if p == "KR":
        return "KR"
    if p == "H":
        return "H"

    return p

def injury_position_report(team_name: str):
    team_df = depth_chart[depth_chart["Team"] == team_name]

    out_report = Counter()
    questionable_report = Counter()

    for _, row in team_df.iterrows():
        status = normalize_status(row.get("Starter"))
        pos = normalize_pos(row.get("Position"))

        if not pos or not status:
            continue

        if status in NO_PLAY:
            out_report[pos] += 1
        elif status in MAYBE_PLAY:
            questionable_report[pos] += 1
        else:
            pass

    return dict(questionable_report), dict(out_report)

# How each unit is affected by injuries at specific positions
POSITION_UNIT_IMPACTS = {
    "pass_offense": {
        "QB": 2.6, "WR": 1.1, "TE": 0.7,
        "LT": 0.9, "LG": 0.6, "C": 0.7, "RG": 0.6, "RT": 0.9,
        "RB": 0.3, "FB": 0.2,
    },
    "run_offense": {
        "RB": 2.0, "FB": 0.7, "TE": 0.6,
        "LT": 1.0, "LG": 0.9, "C": 0.9, "RG": 0.9, "RT": 1.0,
        "QB": 0.2,
    },
    "scoring_offense": {
        "QB": 1.2, "WR": 0.7, "RB": 0.7,
        "LT": 0.5, "LG": 0.4, "C": 0.4, "RG": 0.4, "RT": 0.5,
        "PK": 0.9,
    },
    "situational_offense": {
        "QB": 1.6, "WR": 0.7, "RB": 0.7, "TE": 0.5,
        "LT": 0.7, "LG": 0.5, "C": 0.5, "RG": 0.5, "RT": 0.7,
    },

    "pass_defense": {
        "CB": 1.2, "S": 0.9,
        "DE": 0.6, "DT": 0.5,
        "LB": 0.4,
    },
    "run_defense": {
        "DT": 1.0, "LB": 0.9, "DE": 0.5,
        "CB": 0.2, "S": 0.2,
    },
    "scoring_defense": {
        "CB": 0.8, "S": 0.8,
        "LB": 0.6, "DT": 0.6, "DE": 0.6,
    },
    "situational_defense": {
        "CB": 0.7, "S": 0.7, "LB": 0.7,
        "DT": 0.5, "DE": 0.5,
    },

    "special_teams": {
        "PK": 1.6, "P": 1.2, "LS": 0.8, "PR": 0.9, "KR": 0.9, "H": 0.3,
    }
}

OUT_PENALTY = 0.06
Q_PENALTY = 0.03
MIN_FACTOR = 0.55

def injury_factor_for_unit(team_name: str, unit_key: str) -> float:
    q, o = injury_position_report(team_name)
    impacts = POSITION_UNIT_IMPACTS.get(unit_key, {})

    penalty = 0.0
    for pos, w in impacts.items():
        penalty += w * o.get(pos, 0) * OUT_PENALTY
        penalty += w * q.get(pos, 0) * Q_PENALTY

    factor = 1.0 - penalty
    return max(MIN_FACTOR, min(1.0, factor))

def apply_injury_shrink(score: float, factor: float) -> float:
    if score is None:
        return 0.5
    return 0.5 + (score - 0.5) * factor

# Main metrics

def build_passing_offense_metrics(team_name):
    data = get_team_data('offense', team_name, 'passing')
    if not data:
        q, o = injury_position_report(team_name)
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    att = data.get('Att', 0.0)
    cmp_ = data.get('Cmp', 0.0)
    yds = data.get('Yds', 0.0)
    td = data.get('TD', 0.0)
    inte = data.get('INT', 0.0)
    rate = data.get('Rate', None)
    pass_1st = data.get('Pass 1st', 0.0)
    cmp_pct = data.get('Cmp %', None)
    ypa = data.get('Yds/Att', None)
    twenty_plus = data.get('20+', 0.0)
    forty_plus = data.get('40+', 0.0)
    sck = data.get('Sck', 0.0)
    scky = data.get('SckY', 0.0)
    lng = parse_long_gain(data.get('Lng'))

    td_rate = safe_rate(td, att)
    int_rate = safe_rate(inte, att)
    td_int_ratio = safe_rate(td, inte if inte not in (0, None) else 1)
    first_down_rate = safe_rate(pass_1st, att)
    explosive_20_rate = safe_rate(twenty_plus, att)
    explosive_40_rate = safe_rate(forty_plus, att)

    dropbacks = att + sck
    sack_rate = safe_rate(sck, dropbacks)
    net_ypa = safe_rate(yds - scky, dropbacks) if dropbacks else None

    q, o = injury_position_report(team_name)

    return {
        'team': team_name,
        'has_data': True,
        'attempts': att,
        'completions': cmp_,
        'yards': yds,
        'td': td,
        'int': inte,
        'sacks': sck,
        'sack_yards': scky,
        'pass_first_downs': pass_1st,
        'explosive_20_plus': twenty_plus,
        'explosive_40_plus': forty_plus,
        'cmp_pct': cmp_pct,
        'yards_per_attempt': ypa,
        'passer_rating': rate,
        'longest_completion_yards': lng,
        'td_per_att': td_rate,
        'int_per_att': int_rate,
        'td_int_ratio': td_int_ratio,
        'first_down_per_att': first_down_rate,
        'explosive_20_rate': explosive_20_rate,
        'explosive_40_rate': explosive_40_rate,
        'sack_rate': sack_rate,
        'net_yards_per_attempt': net_ypa,
        'injury_report': {'Q': q, 'O': o}
    }

def build_run_offense_metrics(team_name):
    data = get_team_data('offense', team_name, 'rushing')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    att = data.get('Att', 0.0)
    yds = data.get('Yds', 0.0)
    ypc = data.get('YPC', None)
    rsh_td = data.get('Rsh TD', 0.0)
    rush_1st = data.get('Rush 1st', 0.0)
    rush_1st_pct = data.get('Rush 1st%', None)
    twenty_plus = data.get('20+', 0.0)
    forty_plus = data.get('40+', 0.0)
    lng = parse_long_gain(data.get('Lng'))
    rush_fum = data.get('Rush FUM', 0.0)

    td_per_carry = safe_rate(rsh_td, att)
    first_down_per_carry = safe_rate(rush_1st, att)
    explosive_20_rate = safe_rate(twenty_plus, att)
    explosive_40_rate = safe_rate(forty_plus, att)
    fum_per_carry = safe_rate(rush_fum, att)

    return {
        'team': team_name,
        'has_data': True,
        'attempts': att,
        'yards': yds,
        'rush_td': rsh_td,
        'rush_first_downs': rush_1st,
        'explosive_20_plus': twenty_plus,
        'explosive_40_plus': forty_plus,
        'rush_fumbles': rush_fum,
        'yards_per_carry': ypc,
        'rush_first_down_pct': rush_1st_pct,
        'td_per_carry': td_per_carry,
        'first_down_per_carry': first_down_per_carry,
        'explosive_20_rate': explosive_20_rate,
        'explosive_40_rate': explosive_40_rate,
        'fumbles_per_carry': fum_per_carry,
        'longest_rush_yards': lng,
        'injury_report': {'Q': q, 'O': o}
    }

def build_scoring_offense_metrics(team_name):
    data = get_team_data('offense', team_name, 'scoring')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    tot_td = data.get('Tot TD', 0.0)
    rsh_td = data.get('Rsh TD', 0.0)
    rec_td = data.get('Rec TD', 0.0)
    int_td = data.get('INT TD', 0.0)
    fr_td = data.get('FR TD', 0.0) or 0.0
    fr_td2 = data.get('FR TD.1', 0.0) or 0.0
    two_pt = data.get('2-PT', 0.0)
    scrm_plays = data.get('Scrm Plys', None)

    fg_made = data.get('FGM', 0.0)
    fg_pct = data.get('FG %', None)
    xp_made = data.get('XPM', 0.0)
    xp_pct = data.get('XP Pct', None)

    kret_td = data.get('KRet TD', 0.0)
    pret_td = data.get('PRet T', 0.0)
    non_off_td = int_td + fr_td + fr_td2 + kret_td + pret_td

    td_per_scrimmage = safe_rate(tot_td, scrm_plays)
    two_pt_per_scrimmage = safe_rate(two_pt, scrm_plays)

    return {
        'team': team_name,
        'has_data': True,
        'total_td': tot_td,
        'rush_td': rsh_td,
        'rec_td': rec_td,
        'int_td': int_td,
        'fr_td': fr_td + fr_td2,
        'kret_td': kret_td,
        'pret_td': pret_td,
        'non_off_td': non_off_td,
        'two_pt': two_pt,
        'scrimmage_plays': scrm_plays,
        'td_per_scrimmage_play': td_per_scrimmage,
        'two_pt_per_scrimmage_play': two_pt_per_scrimmage,
        'fg_made': fg_made,
        'fg_pct': fg_pct,
        'xp_made': xp_made,
        'xp_pct': xp_pct,
        'injury_report': {'Q': q, 'O': o}
    }

def build_situational_offense_metrics(team_name):
    data = get_team_data('offense', team_name, 'downs')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    third_att = data.get('3rd Att', 0.0)
    third_md = data.get('3rd Md', 0.0)
    fourth_att = data.get('4th Att', 0.0)
    fourth_md = data.get('4th Md', 0.0)

    third_conv_rate = safe_rate(third_md, third_att)
    fourth_conv_rate = safe_rate(fourth_md, fourth_att)
    total_att = third_att + fourth_att
    total_md = third_md + fourth_md
    total_conv_rate = safe_rate(total_md, total_att)

    return {
        'team': team_name,
        'has_data': True,
        'third_down_attempts': third_att,
        'third_down_made': third_md,
        'third_down_conversion_rate': third_conv_rate,
        'fourth_down_attempts': fourth_att,
        'fourth_down_made': fourth_md,
        'fourth_down_conversion_rate': fourth_conv_rate,
        'total_late_down_attempts': total_att,
        'total_late_down_made': total_md,
        'total_late_down_conversion_rate': total_conv_rate,
        'injury_report': {'Q': q, 'O': o}
    }

def build_run_defense_metrics(team_name):
    data = get_team_data('defense', team_name, 'rushing')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    att = data.get('Att', 0.0)
    yds = data.get('Yds', 0.0)
    ypc = data.get('YPC', None)
    rsh_td = data.get('Rsh TD', 0.0)
    rush_1st = data.get('Rush 1st', 0.0)
    twenty_plus = data.get('20+', 0.0)
    forty_plus = data.get('40+', 0.0)
    lng = parse_long_gain(data.get('Lng'))

    td_per_carry_allowed = safe_rate(rsh_td, att)
    first_down_per_carry_allowed = safe_rate(rush_1st, att)
    explosive_20_rate_allowed = safe_rate(twenty_plus, att)
    explosive_40_rate_allowed = safe_rate(forty_plus, att)

    return {
        'team': team_name,
        'has_data': True,
        'rush_att_allowed': att,
        'rush_yards_allowed': yds,
        'yards_per_carry_allowed': ypc,
        'rush_td_allowed': rsh_td,
        'rush_first_downs_allowed': rush_1st,
        'explosive_20_plus_allowed': twenty_plus,
        'explosive_40_plus_allowed': forty_plus,
        'longest_rush_allowed_yards': lng,
        'td_per_carry_allowed': td_per_carry_allowed,
        'first_down_per_carry_allowed': first_down_per_carry_allowed,
        'explosive_20_rate_allowed': explosive_20_rate_allowed,
        'explosive_40_rate_allowed': explosive_40_rate_allowed,
        'injury_report': {'Q': q, 'O': o}
    }

def build_pass_defense_metrics(team_name):
    data = get_team_data('defense', team_name, 'passing')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    att = data.get('Att', 0.0)
    cmp_ = data.get('Cmp', 0.0)
    yds = data.get('Yds', 0.0)
    td = data.get('TD', 0.0)
    inte = data.get('INT', 0.0)
    rate = data.get('Rate', None)
    pass_1st = data.get('Pass 1st', 0.0)
    cmp_pct = data.get('Cmp %', None)
    ypa = data.get('Yds/Att', None)
    twenty_plus = data.get('20+', 0.0)
    forty_plus = data.get('40+', 0.0)
    sck = data.get('Sck', 0.0)
    scky = data.get('SckY', 0.0)
    lng = parse_long_gain(data.get('Lng'))

    td_rate_allowed = safe_rate(td, att)
    int_per_att = safe_rate(inte, att)
    pass_1st_rate_allowed = safe_rate(pass_1st, att)
    explosive_20_rate_allowed = safe_rate(twenty_plus, att)
    explosive_40_rate_allowed = safe_rate(forty_plus, att)

    dropbacks = att + sck
    sack_rate = safe_rate(sck, dropbacks)
    net_ypa_allowed = safe_rate(yds + scky, dropbacks) if dropbacks else None

    return {
        'team': team_name,
        'has_data': True,
        'pass_att_allowed': att,
        'pass_cmp_allowed': cmp_,
        'pass_yards_allowed': yds,
        'pass_td_allowed': td,
        'ints_made': inte,
        'pass_first_downs_allowed': pass_1st,
        'explosive_20_plus_allowed': twenty_plus,
        'explosive_40_plus_allowed': forty_plus,
        'sacks_made': sck,
        'sack_yards': scky,
        'longest_completion_allowed_yards': lng,
        'cmp_pct_allowed': cmp_pct,
        'yards_per_attempt_allowed': ypa,
        'passer_rating_allowed': rate,
        'td_per_att_allowed': td_rate_allowed,
        'int_per_att_defense': int_per_att,
        'pass_first_down_rate_allowed': pass_1st_rate_allowed,
        'explosive_20_rate_allowed': explosive_20_rate_allowed,
        'explosive_40_rate_allowed': explosive_40_rate_allowed,
        'sack_rate': sack_rate,
        'net_yards_per_attempt_allowed': net_ypa_allowed,
        'injury_report': {'Q': q, 'O': o}
    }

def build_scoring_defense_metrics(team_name):
    data = get_team_data('defense', team_name, 'scoring')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    tot_td = (data.get('Tot TD', 0.0) or 0.0)
    rsh_td = (data.get('Rsh TD', 0.0) or 0.0)
    rec_td = (data.get('Rec TD', 0.0) or 0.0)
    int_td = (data.get('INT TD', 0.0) or 0.0)

    fr_td  = (data.get('FR TD', 0.0) or 0.0) + (data.get('FR TD.1', 0.0) or 0.0)

    two_pt = (data.get('2-PT', 0.0) or 0.0)
    scrm_plays = (data.get('Scrm Plys', 0.0) or 0.0)

    fg_made_allowed = (data.get('FGM', 0.0) or 0.0)
    fg_pct_allowed  = data.get('FG %', None)
    xp_made_allowed = (data.get('XPM', 0.0) or 0.0)
    xp_pct_allowed  = data.get('XP Pct', None)

    kret_td_allowed = (data.get('KRet TD', 0.0) or 0.0)
    pret_td_allowed = (data.get('PRet T', 0.0) or 0.0)

    td_per_scrimmage_allowed = safe_rate(tot_td, scrm_plays)

    return {
        'team': team_name,
        'has_data': True,
        'total_td_allowed': tot_td,
        'rush_td_allowed': rsh_td,
        'rec_td_allowed': rec_td,
        'int_td_allowed': int_td,
        'fr_td_allowed': fr_td,
        'kick_return_td_allowed': kret_td_allowed,
        'punt_return_td_allowed': pret_td_allowed,
        'two_pt_allowed': two_pt,
        'scrimmage_plays_defended': scrm_plays,
        'td_per_scrimmage_play_allowed': td_per_scrimmage_allowed,
        'fg_made_allowed': fg_made_allowed,
        'xp_made_allowed': xp_made_allowed,
        'fg_pct_allowed': fg_pct_allowed,
        'xp_pct_allowed': xp_pct_allowed,
        'injury_report': {'Q': q, 'O': o}
    }

def build_situational_defense_metrics(team_name):
    data = get_team_data('defense', team_name, 'downs')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    third_att = data.get('3rd Att', 0.0)
    third_md = data.get('3rd Md', 0.0)
    fourth_att = data.get('4th Att', 0.0)
    fourth_md = data.get('4th Md', 0.0)

    third_conv_allowed = safe_rate(third_md, third_att)
    fourth_conv_allowed = safe_rate(fourth_md, fourth_att)
    total_att = third_att + fourth_att
    total_md = third_md + fourth_md
    total_conv_allowed = safe_rate(total_md, total_att)

    return {
        'team': team_name,
        'has_data': True,
        'third_down_att_faced': third_att,
        'third_down_conv_allowed': third_md,
        'third_down_conversion_rate_allowed': third_conv_allowed,
        'fourth_down_att_faced': fourth_att,
        'fourth_down_conv_allowed': fourth_md,
        'fourth_down_conversion_rate_allowed': fourth_conv_allowed,
        'late_down_att_faced': total_att,
        'late_down_conv_allowed': total_md,
        'late_down_conversion_rate_allowed': total_conv_allowed,
        'injury_report': {'Q': q, 'O': o}
    }

def build_special_team_metrics(team_name):
    data = get_team_data('special', team_name, 'scoring')
    q, o = injury_position_report(team_name)

    if not data:
        return {'team': team_name, 'has_data': False, 'injury_report': {'Q': q, 'O': o}}

    fg_made = data.get('FGM', 0.0)
    fg_pct = data.get('FG %', None)
    xp_made = data.get('XPM', 0.0)
    xp_pct = data.get('XP Pct', None)
    kret_td = data.get('KRet TD', 0.0)
    pret_td = data.get('PRet T', 0.0)
    tot_td = data.get('Tot TD', 0.0)

    return {
        'team': team_name,
        'has_data': True,
        'fg_made': fg_made,
        'fg_pct': fg_pct,
        'xp_made': xp_made,
        'xp_pct': xp_pct,
        'kick_return_td': kret_td,
        'punt_return_td': pret_td,
        'total_special_td': tot_td,
        'injury_report': {'Q': q, 'O': o}
    }

# Scoring / normalization

def _normalize_pair(v1, v2, higher_better=True, k=3.0):
    try:
        v1 = float(v1)
        v2 = float(v2)
    except (TypeError, ValueError):
        return 0.5, 0.5

    if math.isclose(v1, v2, rel_tol=1e-8, abs_tol=1e-8):
        return 0.5, 0.5

    if not higher_better:
        v1, v2 = -v1, -v2

    scale = max(abs(v1), abs(v2), 1e-6)
    x = (v1 - v2) / scale

    p1 = 1.0 / (1.0 + math.exp(-k * x))
    p2 = 1.0 - p1
    return p1, p2

def _unit_score_from_metrics(m1, m2, config, default_if_missing=0.5):
    has1 = m1.get('has_data', False)
    has2 = m2.get('has_data', False)

    if not has1 and not has2:
        return default_if_missing, default_if_missing
    if has1 and not has2:
        return default_if_missing, default_if_missing
    if has2 and not has1:
        return default_if_missing, default_if_missing

    weighted1 = 0.0
    weighted2 = 0.0
    total_weight = 0.0

    for key, higher_better, weight in config:
        v1 = m1.get(key, None)
        v2 = m2.get(key, None)

        if v1 is None or v2 is None:
            continue
        if isinstance(v1, float) and math.isnan(v1):
            continue
        if isinstance(v2, float) and math.isnan(v2):
            continue

        s1, s2 = _normalize_pair(v1, v2, higher_better=higher_better, k=3.0)
        weighted1 += s1 * weight
        weighted2 += s2 * weight
        total_weight += weight

    if total_weight == 0:
        return default_if_missing, default_if_missing

    return weighted1 / total_weight, weighted2 / total_weight

def home_field_scores(team1, team2):
    t1_home = team1 in home_field and team2 not in home_field
    t2_home = team2 in home_field and team1 not in home_field

    if t1_home and not t2_home:
        return 0.55, 0.45
    elif t2_home and not t1_home:
        return 0.45, 0.55
    else:
        return 0.5, 0.5

def power_rank_scores(team1, team2):
    pr1 = power_ranking(team1)
    pr2 = power_ranking(team2)
    s1, s2 = _normalize_pair(pr1, pr2, higher_better=True, k=1.5)
    return s1, s2

# Presenting metrics

PASS_OFFENSE_CONFIG = [
    ('yards_per_attempt', True, 0.30),
    ('td_per_att', True, 0.30),
    ('int_per_att', False, 0.20),
    ('first_down_per_att', True, 0.20),
]

RUN_OFFENSE_CONFIG = [
    ('yards_per_carry', True, 0.35),
    ('td_per_carry', True, 0.30),
    ('first_down_per_carry', True, 0.20),
    ('fumbles_per_carry', False, 0.15),
]

SCORING_OFFENSE_CONFIG = [
    ('total_td',   True, 0.50),
    ('fg_made',    True, 0.25),
    ('xp_made',    True, 0.25),
]

SITUATIONAL_OFFENSE_CONFIG = [
    ('third_down_conversion_rate', True, 0.45),
    ('fourth_down_conversion_rate', True, 0.30),
    ('total_late_down_conversion_rate', True, 0.25),
]

RUN_DEFENSE_CONFIG = [
    ('yards_per_carry_allowed', False, 0.40),
    ('td_per_carry_allowed', False, 0.30),
    ('first_down_per_carry_allowed', False, 0.30),
]

PASS_DEFENSE_CONFIG = [
    ('yards_per_attempt_allowed', False, 0.30),
    ('td_per_att_allowed', False, 0.30),
    ('int_per_att_defense', True, 0.20),
    ('pass_first_down_rate_allowed', False, 0.20),
]

SCORING_DEFENSE_CONFIG = [
    ('total_td_allowed', False, 0.50),
    ('fg_made_allowed',  False, 0.25),
    ('xp_made_allowed',  False, 0.25),
]

SITUATIONAL_DEFENSE_CONFIG = [
    ('third_down_conversion_rate_allowed', False, 0.45),
    ('fourth_down_conversion_rate_allowed', False, 0.30),
    ('late_down_conversion_rate_allowed', False, 0.25),
]

SPECIAL_TEAMS_CONFIG = [
    ('fg_pct', True, 0.35),
    ('xp_pct', True, 0.25),
    ('kick_return_td', True, 0.20),
    ('punt_return_td', True, 0.20),
]

# Unit rating functions

def pass_offense_rating(team1, team2):
    m1 = build_passing_offense_metrics(team1)
    m2 = build_passing_offense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, PASS_OFFENSE_CONFIG)
    return {team1: s1, team2: s2}

def run_offense_rating(team1, team2):
    m1 = build_run_offense_metrics(team1)
    m2 = build_run_offense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, RUN_OFFENSE_CONFIG)
    return {team1: s1, team2: s2}

def scoring_offense_rating(team1, team2):
    m1 = build_scoring_offense_metrics(team1)
    m2 = build_scoring_offense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, SCORING_OFFENSE_CONFIG)
    return {team1: s1, team2: s2}

def situational_offense_rating(team1, team2):
    m1 = build_situational_offense_metrics(team1)
    m2 = build_situational_offense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, SITUATIONAL_OFFENSE_CONFIG)
    return {team1: s1, team2: s2}

def run_defense_rating(team1, team2):
    m1 = build_run_defense_metrics(team1)
    m2 = build_run_defense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, RUN_DEFENSE_CONFIG)
    return {team1: s1, team2: s2}

def pass_defense_rating(team1, team2):
    m1 = build_pass_defense_metrics(team1)
    m2 = build_pass_defense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, PASS_DEFENSE_CONFIG)
    return {team1: s1, team2: s2}

def scoring_defense_rating(team1, team2):
    m1 = build_scoring_defense_metrics(team1)
    m2 = build_scoring_defense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, SCORING_DEFENSE_CONFIG)
    return {team1: s1, team2: s2}

def situational_defense_rating(team1, team2):
    m1 = build_situational_defense_metrics(team1)
    m2 = build_situational_defense_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, SITUATIONAL_DEFENSE_CONFIG)
    return {team1: s1, team2: s2}

def special_team_rating(team1, team2):
    m1 = build_special_team_metrics(team1)
    m2 = build_special_team_metrics(team2)
    s1, s2 = _unit_score_from_metrics(m1, m2, SPECIAL_TEAMS_CONFIG)
    return {team1: s1, team2: s2}

# Compare Team Matchups

def compare(team1, team2):
    pr1, pr2 = power_rank_scores(team1, team2)
    hf1, hf2 = home_field_scores(team1, team2)

    pass_off = pass_offense_rating(team1, team2)
    run_off = run_offense_rating(team1, team2)
    score_off = scoring_offense_rating(team1, team2)
    situ_off = situational_offense_rating(team1, team2)

    run_def = run_defense_rating(team1, team2)
    pass_def = pass_defense_rating(team1, team2)
    score_def = scoring_defense_rating(team1, team2)
    situ_def = situational_defense_rating(team1, team2)

    special = special_team_rating(team1, team2)

    unit_scores = {
        'pass_offense': pass_off,
        'run_offense': run_off,
        'scoring_offense': score_off,
        'situational_offense': situ_off,
        'run_defense': run_def,
        'pass_defense': pass_def,
        'scoring_defense': score_def,
        'situational_defense': situ_def,
        'special_teams': special,
    }

    for unit_key, score_dict in unit_scores.items():
        f1 = injury_factor_for_unit(team1, unit_key)
        f2 = injury_factor_for_unit(team2, unit_key)
        score_dict[team1] = apply_injury_shrink(score_dict[team1], f1)
        score_dict[team2] = apply_injury_shrink(score_dict[team2], f2)

    WEIGHTS = {
        'power_rank': 0.05,
        'home_field': 0.03,
        'pass_offense': 0.18,
        'run_offense': 0.12,
        'pass_defense': 0.16,
        'run_defense': 0.12,
        'situational_offense': 0.08,
        'situational_defense': 0.08,
        'scoring_offense': 0.07,
        'scoring_defense': 0.07,
        'special_teams': 0.04,
    }

    t1, t2 = team1, team2

    total1 = (
        WEIGHTS['power_rank'] * pr1 +
        WEIGHTS['home_field'] * hf1 +
        WEIGHTS['pass_offense'] * pass_off[t1] +
        WEIGHTS['run_offense'] * run_off[t1] +
        WEIGHTS['scoring_offense'] * score_off[t1] +
        WEIGHTS['situational_offense'] * situ_off[t1] +
        WEIGHTS['pass_defense'] * pass_def[t1] +
        WEIGHTS['run_defense'] * run_def[t1] +
        WEIGHTS['scoring_defense'] * score_def[t1] +
        WEIGHTS['situational_defense'] * situ_def[t1] +
        WEIGHTS['special_teams'] * special[t1]
    )

    total2 = (
        WEIGHTS['power_rank'] * pr2 +
        WEIGHTS['home_field'] * hf2 +
        WEIGHTS['pass_offense'] * pass_off[t2] +
        WEIGHTS['run_offense'] * run_off[t2] +
        WEIGHTS['scoring_offense'] * score_off[t2] +
        WEIGHTS['situational_offense'] * situ_off[t2] +
        WEIGHTS['pass_defense'] * pass_def[t2] +
        WEIGHTS['run_defense'] * run_def[t2] +
        WEIGHTS['scoring_defense'] * score_def[t2] +
        WEIGHTS['situational_defense'] * situ_def[t2] +
        WEIGHTS['special_teams'] * special[t2]
    )

    predicted = t1 if total1 > total2 else t2

    return {
        'teams': [t1, t2],
        'power_rank_scores': {t1: pr1, t2: pr2},
        'home_field_scores': {t1: hf1, t2: hf2},
        'unit_scores': unit_scores,
        'total_scores': {t1: total1, t2: total2},
        'predicted_winner': predicted,
    }

# Plotting

def plot_unit_scores(result, color1, color2):
    team1, team2 = result['teams']
    unit_scores = result['unit_scores']

    units = list(unit_scores.keys())
    t1_vals = [unit_scores[u][team1] for u in units]
    t2_vals = [unit_scores[u][team2] for u in units]

    x = range(len(units))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], t1_vals, width, label=team1, color=color1)
    plt.bar([i + width / 2 for i in x], t2_vals, width, label=team2, color=color2)
    plt.xticks(list(x), units, rotation=45, ha='right')
    plt.ylabel('Score (0–1)')
    plt.title(f'Unit Comparison: {team1} vs {team2}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_total_scores(result, color1, color2):
    team1, team2 = result['teams']
    totals = result['total_scores']

    teams = [team1, team2]
    values = [totals[team1], totals[team2]]
    x = range(len(teams))

    plt.figure()
    plt.bar(x, values, color=[color1, color2])
    plt.xticks(list(x), teams)
    plt.ylabel('Total Score (0–1)')
    plt.title('Overall Matchup Score')
    plt.tight_layout()
    plt.show()

def plot_offense_radar(team1, team2, color1, color2):
    result = compare(team1, team2)
    unit_scores = result['unit_scores']

    labels = ['pass_offense', 'run_offense', 'scoring_offense', 'situational_offense']
    n_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    t1_vals = [unit_scores[label][team1] for label in labels]
    t2_vals = [unit_scores[label][team2] for label in labels]

    t1_vals = np.concatenate((t1_vals, [t1_vals[0]]))
    t2_vals = np.concatenate((t2_vals, [t2_vals[0]]))

    plt.figure()
    ax = plt.subplot(projection='polar')
    ax.plot(angles, t1_vals, marker='o', label=team1, color=color1)
    ax.fill(angles, t1_vals, alpha=0.1, color=color1)

    ax.plot(angles, t2_vals, marker='o', label=team2, color=color2)
    ax.fill(angles, t2_vals, alpha=0.1, color=color2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Pass O', 'Run O', 'Score O', 'Situ O'])
    ax.set_ylim(0, 1)
    ax.set_title(f'Offensive Comparison: {team1} vs {team2}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()

def plot_defense_radar(team1, team2, color1, color2):
    result = compare(team1, team2)
    unit_scores = result['unit_scores']

    labels = ['pass_defense', 'run_defense', 'scoring_defense', 'situational_defense']
    n_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    t1_vals = [unit_scores[label][team1] for label in labels]
    t2_vals = [unit_scores[label][team2] for label in labels]

    t1_vals = np.concatenate((t1_vals, [t1_vals[0]]))
    t2_vals = np.concatenate((t2_vals, [t2_vals[0]]))

    plt.figure()
    ax = plt.subplot(projection='polar')
    ax.plot(angles, t1_vals, marker='o', label=team1, color=color1)
    ax.fill(angles, t1_vals, alpha=0.1, color=color1)

    ax.plot(angles, t2_vals, marker='o', label=team2, color=color2)
    ax.fill(angles, t2_vals, alpha=0.1, color=color2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Pass D', 'Run D', 'Score D', 'Situ D'])
    ax.set_ylim(0, 1)
    ax.set_title(f'Defensive Comparison: {team1} vs {team2}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()

# Team colors + Main function

team_colors = {
    "Cardinals": "#97233F", "Falcons": "#A71930", "Ravens": "#241773",
    "Bills": "#00338D", "Panthers": "#0085CA", "Bears": "#0B162A",
    "Bengals": "#FB4F14", "Browns": "#311D00", "Cowboys": "#041E42",
    "Broncos": "#002244", "Lions": "#0076B6", "Packers": "#203731",
    "Texans": "#03202F", "Colts": "#003DA5", "Jaguars": "#006778",
    "Chiefs": "#E31837", "Raiders": "#000000", "Chargers": "#0080C6",
    "Rams": "#003594", "Dolphins": "#008E97", "Vikings": "#4F2683",
    "Patriots": "#002244", "Saints": "#D3BC8D", "Giants": "#0B2265",
    "Jets": "#125740", "Eagles": "#004C54", "Steelers": "#FFB612",
    "Seahawks": "#002244", "49ers": "#AA0000", "Buccaneers": "#D50A0A",
    "Titans": "#4B92DB", "Commanders": "#5A1414",
}

def compare_and_plot(team1, team2):
    color1 = team_colors.get(team1, "#444444")
    color2 = team_colors.get(team2, "#888888")

    result = compare(team1, team2)
    print("Predicted winner:", result['predicted_winner'])
    print("Total scores:", result['total_scores'])

    plot_unit_scores(result, color1, color2)
    plot_total_scores(result, color1, color2)
    plot_offense_radar(team1, team2, color1, color2)
    plot_defense_radar(team1, team2, color1, color2)

    return result

if __name__ == "__main__":
    for t1, t2 in matchup:
        final = compare_and_plot(t1, t2)
        pprint(final)
