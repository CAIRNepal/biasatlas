"""Shared institution-name canonicalization used by both the Fig 4
university heatmap scripts and statistical_tests.py's institutional Gini,
so the two report consistent institution counts / identities."""
import re
import pandas as pd


def basic_affil_preclean(name: str) -> str:
    if pd.isna(name) or not str(name).strip():
        return ""
    n = str(name).strip()
    n = re.sub(
        r"\s*\((united states|usa|uk|canada|germany|australia|spain|europe|european|global)\)\s*$",
        "", n, flags=re.I
    )
    n = n.replace("’", "'").replace("–", "-").replace("—", "-")
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"[,\.;:\s]+$", "", n)
    return n.strip()


# --- FULL Canonical mapping (regex -> canonical label). ---
CANONICAL_ORG_MAP = {
    # --- Major US/UK/EU universities ---
    r"\bmit\b|massachusetts institute of technology": "MIT",
    r"\bharvard\b": "Harvard University",
    r"\bstanford\b": "Stanford University",
    r"\buniversity college london\b|\bucl\b": "University College London",
    r"\brutgers\b(?!\s*business)|rutgers,\s*the\s*state\s*university\s*of\s*new\s*jersey": "Rutgers University",
    r"\buniversity of pennsylvania\b|\bupenn\b": "University of Pennsylvania",
    r"\buniversity of michigan\b": "University of Michigan",
    r"\bmichigan state\b": "Michigan State University",
    r"\bnorth carolina state\b": "North Carolina State University",
    r"\bcarnegie mellon\b|\bcmu\b": "Carnegie Mellon University",
    r"\bcolumbia\b": "Columbia University",
    r"\bcornell\b": "Cornell University",
    r"\byale\b": "Yale University",
    r"\buniversity of washington(?!\s*bothell)": "University of Washington",
    r"\buniversity of washington bothell\b": "University of Washington Bothell",
    r"\buniversity of southern california\b|\busc\b": "University of Southern California",
    r"\buniversity of virginia\b": "University of Virginia",
    r"\buniversity of arizona\b": "University of Arizona",
    r"\buniversity of utah\b": "University of Utah",
    r"\buniversity of georgia\b": "University of Georgia",
    r"\buniversity of illinois chicago\b": "University of Illinois Chicago",
    r"\buniversity of minnesota(?!, duluth)": "University of Minnesota",
    r"\buniversity of minnesota,\s*duluth\b": "University of Minnesota Duluth",
    r"\bqueen'?s university\b": "Queen's University",
    r"\bgeorge washington university\b": "George Washington University",

    # --- UC specifics ---
    r"\bucla\b|\buniversity of california,\s*los angeles\b": "UCLA",
    r"\buc berkeley\b|\buniversity of california,\s*berkeley\b": "UC Berkeley",
    r"\buniversity of california,\s*riverside\b": "UC Riverside",

    # --- Europe (UK/EU) ---
    r"\buniversity of oxford\b|\boxford university\b|\boxford\b": "University of Oxford",
    r"\buniversity of cambridge\b|\bcambridge university\b|\bcambridge\b": "University of Cambridge",
    r"\bimperial college london\b": "Imperial College London",
    r"\brwth aachen\b": "RWTH Aachen University",
    r"\btechnische universität berlin\b|\btu berlin\b": "Technische Universität Berlin",
    r"\bcharit[eé]\b.*universitätsmedizin berlin": "Charité - Universitätsmedizin Berlin",
    r"\buniversity college dublin\b": "University College Dublin",
    r"\bthe alan turing institute\b": "The Alan Turing Institute",
    r"\btib leibniz information centre.*\b": "TIB Leibniz Information Centre for Science and Technology",
    r"\bl3s research center.*leibniz university hannover\b": "L3S Research Center & Leibniz University Hannover",
    r"\bipvs, universität stuttgart\b|\buniversität stuttgart\b": "University of Stuttgart",
    r"\bgesis\b.*social sciences": "GESIS Leibniz Institute for the Social Sciences",
    r"\b[ée]cole polytechnique\b": "École Polytechnique",
    r"\bvienna university of economics and business\b": "Vienna University of Economics and Business",
    r"\buniversity of portsmouth\b": "University of Portsmouth",
    r"\bleiden university medical center\b": "Leiden University Medical Center",
    r"\buniversity hospital of zurich\b": "University Hospital of Zurich",
    r"\buniversity of zurich\b": "University of Zurich",
    r"\bzurich university of applied sciences\b": "Zurich University of Applied Sciences",
    r"\bheinrich-heine-universit[aä]t d[üu]sseldorf\b": "Heinrich-Heine-Universität Düsseldorf",
    r"\binstitute for legal informatics, leibniz university of hanover\b": "Leibniz University Hannover",
    r"\binstitute of computer science, forth-ics\b": "FORTH-ICS",
    r"\binnovation lab, schu[aä]fa holding ag\b": "SCHUFA Holding AG",
    r"\binformation technologies institute, certh\b": "CERTH – Information Technologies Institute",
    r"\buniversit[eé] de toulouse\b|\binstitut de math[eé]matiques de toulouse\b": "Université de Toulouse",
    r"\buniversity of southampton\b|\buniversity of southampton.*electronics and computer science\b": "University of Southampton",

    # --- Asia (CN/HK/TW/KR) ---
    r"\btsinghua university\b": "Tsinghua University",
    r"\bpeking university\b": "Peking University",
    r"\bfudan university\b": "Fudan University",
    r"\bshanghai jiao tong university\b": "Shanghai Jiao Tong University",
    r"\bzhejiang university\b": "Zhejiang University",
    r"\bnational taiwan university\b": "National Taiwan University",
    r"\buniversity of hong kong\b": "University of Hong Kong",
    r"\bhong kong university of science and technology \((guangzhou|gz)\)|\bhkust\b|\bthe hong kong university of science and technology\b":
        "The Hong Kong University of Science and Technology",
    r"\bthe hong kong polytechnic university\b": "The Hong Kong Polytechnic University",
    r"\bcity university of hong kong\b": "City University of Hong Kong",
    r"\bzhongguancun laboratory\b": "Zhongguancun Laboratory",
    r"\bzhongguancun laboratory, tsinghua university\b": "Zhongguancun Laboratory",
    r"\bharbin institute of technology, shenzhen\b": "Harbin Institute of Technology, Shenzhen",
    r"\bmoe key laboratory of high confidence software technologies\b": "MOE Key Laboratory of High Confidence Software Technologies",
    r"\bwestlake university\b": "Westlake University",
    r"\byen[gk]se?i university\b": "Yonsei University",
    r"\bkaist\b|\bkorea advanced institute of science and technology\b": "KAIST",
    r"\bjilin university\b": "Jilin University",
    r"\brenmin university\b": "Renmin University of China",

    # --- Australia / NZ ---
    r"\bunsw sydney\b": "UNSW Sydney",
    r"\buniversity of technology sydney\b": "University of Technology Sydney",
    r"\buniversity of wollongong\b": "University of Wollongong",
    r"\bswansea university\b": "Swansea University",  # note: had AU mark in CSV once
    r"\bunited arab emirates university\b": "United Arab Emirates University",

    # --- Hospitals / Clinics / Health orgs ---
    r"\bbrigham and women'?s hospital\b": "Brigham and Women's Hospital",
    r"\bmassachusetts general hospital\b": "Massachusetts General Hospital",
    r"\bnorthwestern university\b": "Northwestern University",
    r"\bmayo clinic\b": "Mayo Clinic",
    r"\bmedical university of graz\b": "Medical University of Graz",
    r"\buniversity hospital carl gustav carus\b": "University Hospital Carl Gustav Carus",
    r"\bcenter for devices and radiological health\b": "Center for Devices and Radiological Health",
    r"\bfraunhofer institute for digital medicine\b": "Fraunhofer Institute for Digital Medicine",
    r"\btampere university\b": "Tampere University",
    r"\bst\.?\s*helena hospital\b": "St. Helena Hospital",
    r"\bst\.?\s*luke'?s international university\b": "St. Luke's International University",
    r"\bkaiser permanente\b": "Kaiser Permanente",

    # --- Companies / Labs ---
    r"\bgoogle research\b": "Google Research",
    r"\bgoogle\b": "Google",
    r"\bibm research - india\b": "IBM Research India",
    r"\bibm research\b": "IBM Research",
    r"\bmicrosoft research\b": "Microsoft Research",
    r"\bmicrosoft \((united states|canada)\)\b": "Microsoft",
    r"\bamazon aws ai\b": "Amazon AWS AI",
    r"\bamazon\b": "Amazon",
    r"\badobe research\b": "Adobe Research",
    r"\badobe systems\b": "Adobe",
    r"\bdeepmind\b": "DeepMind",
    r"\balibaba\b": "Alibaba",
    r"\bant group\b": "Ant Group",
    r"\bintel\b": "Intel",
    r"\btwitter\b": "Twitter",
    r"\btencent\b": "Tencent",
    r"\bhuawei noah'?s ark lab\b|\bhuawei\b": "Huawei",
    r"\barista\b": "Arista",
    r"\bolympus\b": "Olympus",
    r"\bopen knowledge\b": "Open Knowledge Foundation",

    # --- Misc / Institutes / Faculties ---
    r"\bknowledge media institute, the open university\b|\bthe open university\b": "The Open University",
    r"\blaboratoire hubert curien\b": "Laboratoire Hubert Curien",
    r"\binstitut polytechnique de paris\b": "Institut Polytechnique de Paris",
    r"\bamsterdam university medical centers\b|\bamsterdam umc\b": "Amsterdam University Medical Centers",
    r"\bdepartment of cognitive and brain sciences, hebrew university\b": "Hebrew University of Jerusalem",
    r"\bfedermann center.*hebrew university\b": "Hebrew University of Jerusalem",
    r"\bfaculty of data and decision sciences, technion\b|\btechnion\b": "Technion – Israel Institute of Technology",
    r"\bkddlab,? dipartimento di informatica, universit[aà] di pisa\b|\buniversity of pisa\b": "University of Pisa",
    r"\binstitut[o|e] de matem[aá]ticas.*valladolid\b|\buniversity of valladolid\b": "University of Valladolid",
    r"\binstitut[o|e] de matem[aá]ticas.*toulouse\b": "Université de Toulouse",
    r"\baalborg university\b": "Aalborg University",
    r"\bhaverford college\b": "Haverford College",
    r"\bwestern university\b": "Western University",
    r"\buniversity of rochester\b": "University of Rochester",
    r"\buniversity of cagliari\b": "University of Cagliari",
    r"\bmacquarie university\b": "Macquarie University",
    r"\by[ıi]ld[ıi]z technical university\b": "Yıldız Technical University",
    r"\bbennett university\b": "Bennett University",
    r"\bcmr university\b": "CMR University",
    r"\bcape breton university\b": "Cape Breton University",
    r"\bcardiovascular institute of the south\b": "Cardiovascular Institute of the South",
    r"\bthe chinese university of hong kong\b": "The Chinese University of Hong Kong",
    r"\buniversity of central florida\b": "University of Central Florida",
    r"\buniversity of notre dame\b": "University of Notre Dame",
    r"\buniversity of padua\b": "University of Padua",
    r"\buniversity of pennsylvania.*health": "University of Pennsylvania",
    r"\buniversity of waterloo\b": "University of Waterloo",
    r"\buniversity of victoria\b": "University of Victoria",
}
COMPILED_MAP = [(re.compile(pat, flags=re.I), canon) for pat, canon in CANONICAL_ORG_MAP.items()]


def canonicalize_org(name: str) -> str | None:
    if pd.isna(name) or not str(name).strip():
        return None
    pre = basic_affil_preclean(name)
    low = pre.lower()
    for rx, canon in COMPILED_MAP:
        if rx.search(low):
            return canon
    return pre  # fallback to pre-cleaned
