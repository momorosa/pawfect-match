def _first_str(val):
    """Return the first non-empty string from val (string or iterable), else None."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    if isinstance(val, (list, tuple, set)):
        for x in val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                return s
    # fallback
    s = str(val).strip()
    return s if s else None

def size_to_weight_filter(size):
    s = _first_str(size)
    if not s:
        return None
    s = s.lower()
    if s in ["x-small", "extra small", "toy"]:
        return {"max_weight": {"$lte": 4.5}}
    if s == "small":
        return {"max_weight": {"$lte": 11.5}}
    if s == "medium":
        return {"max_weight": {"$gte": 11.5, "$lte": 22.7}}
    if s == "large":
        return {"max_weight": {"$gte": 22.7, "$lte": 40.8}}
    if s in ["x-large", "extra large", "giant"]:
        return {"max_weight": {"$gte": 40.8}}
    return None


def clamp01(x):
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return None

def _to_str_list(val):
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip().lower()
        return [s] if s else []
    if isinstance(val, (list, tuple, set)):
        out = []
        for x in val:
            if x is None: 
                continue
            s = str(x).strip().lower()
            if s:
                out.append(s)
        return out
    return [str(val).strip().lower()]

def build_pinecone_filter(c: dict) -> dict:
    flt = {}

    # size or explicit weight
    sz = size_to_weight_filter(c.get("size"))
    if sz:
        flt.update(sz)

    wmin = c.get("weight_min_kg")
    wmax = c.get("weight_max_kg")
    if wmin is not None or wmax is not None:
        if wmin is not None and wmax is not None:
            flt["max_weight"] = {"$gte": float(wmin), "$lte": float(wmax)}
        elif wmin is not None:
            flt["max_weight"] = {"$gte": float(wmin)}
        else:
            flt["max_weight"] = {"$lte": float(wmax)}

    # numeric scales 0..1
    smax, smin = clamp01(c.get("shedding_max")), clamp01(c.get("shedding_min"))
    if smax is not None or smin is not None:
        if smax is not None and smin is not None:
            flt["shedding_value"] = {"$gte": smin, "$lte": smax}
        elif smax is not None:
            flt["shedding_value"] = {"$lte": smax}
        else:
            flt["shedding_value"] = {"$gte": smin}

    emax, emin = clamp01(c.get("energy_max")), clamp01(c.get("energy_min"))
    if emax is not None or emin is not None:
        if emax is not None and emin is not None:
            flt["energy_level_value"] = {"$gte": emin, "$lte": emax}
        elif emax is not None:
            flt["energy_level_value"] = {"$lte": emax}
        else:
            flt["energy_level_value"] = {"$gte": emin}

    tmin = clamp01(c.get("trainability_min"))
    if tmin is not None:
        flt["trainability_value"] = {"$gte": tmin}

    gmax = clamp01(c.get("grooming_max"))
    if gmax is not None:
        flt["grooming_frequency_value"] = {"$lte": gmax}

    # list fields (normalize once)
    colors = _to_str_list(c.get("colors"))
    if colors:
        flt["colors"] = {"$in": colors}

    ears = _to_str_list(c.get("ear_shape"))
    if ears:
        # optional: normalize synonyms (keeps your parser flexible)
        norm = []
        for e in ears:
            if e == "pointed":
                norm.extend(["pointed", "erect"])
            else:
                norm.append(e)
        flt["ear_shape"] = {"$in": list(dict.fromkeys(norm))}  # de-dupe, keep order

    coats = _to_str_list(c.get("coat_texture"))
    if coats:
        flt["coat_texture"] = {"$in": coats}

    return flt or None
