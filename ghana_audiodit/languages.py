"""The 43 languages the model was fine-tuned on.

Keys are the dataset configuration names (``<Name>_<iso639-3>``); ``code`` is the
africa-g2p language code used for universal-spelling conversion.
"""

_KEYS = [
    "Akuapem_Twi_twi", "Anyin_any", "Asante_Twi_twi", "Avatime_avn", "Bassar_Ntcham_bud", "Bimoba_bim",
    "Birifor_Southern_biv", "Bissa_bib", "Buli_bwu", "Chumburung_ncu", "Dagaare_dga", "Dagbani_dag",
    "Dangme_ada", "Deg_mzw", "English_eng", "Ewe_ewe", "Fante_fat", "Fulfulde_Maasina_ffm", "Gikyode_acd",
    "Gonja_gjn", "Hausa_hau", "Kabiye_kbp", "Kasem_xsm", "Konkomba_xon", "Konni_kma", "Kusaal_kus",
    "Lelemi_lef", "Mampruli_maw", "Nawuri_naw", "Ninkare_gur", "Nkonya_nko", "Ntrubo_ntr", "Nzema_nzi",
    "Paasaal_sig", "Sehwi_sfw", "Sekpele_lip", "Selee_snw", "Sisaala_Tumulung_sil", "Siwu_akp",
    "Tampulma_tpm", "Tem_kdh", "Tuwuli_bov", "Vagla_vag",
]

_DISPLAY = {
    "Bassar_Ntcham_bud": "Ntcham (Bassar)",
    "Birifor_Southern_biv": "Southern Birifor",
    "English_eng": "English (Ghanaian)",
    "Fulfulde_Maasina_ffm": "Fulfulde (Maasina)",
    "Ninkare_gur": "Ninkare (Gurenɛ)",
    "Sisaala_Tumulung_sil": "Sisaala (Tumulung)",
}

LANGUAGES: dict[str, dict[str, str]] = {
    k: {"name": _DISPLAY.get(k, k.rsplit("_", 1)[0].replace("_", " ")), "code": k.rsplit("_", 1)[1]}
    for k in _KEYS
}


def resolve(language: str) -> str:
    """Accept a key ('Asante_Twi_twi'), an ISO code ('ewe') or a display name ('Asante Twi')."""
    if language in LANGUAGES:
        return language
    low = language.strip().lower()
    for k, v in LANGUAGES.items():
        if low in (v["name"].lower(), k.lower()):
            return k
    matches = [k for k, v in LANGUAGES.items() if v["code"] == low]
    if len(matches) == 1:
        return matches[0]
    if low == "twi":                     # two Twi varieties share the code
        return "Asante_Twi_twi"
    raise ValueError(f"unknown language {language!r}; choose from: {', '.join(LANGUAGES)}")
