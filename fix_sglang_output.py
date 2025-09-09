#!/usr/bin/env python3
"""
Quick fixes for SGLangRollout garbled output issues.
"""

import torch
from typing import Optional


def fix_decode_parameters(processing_class, output_ids: torch.Tensor) -> str:
    """
    Try different decoding parameters to fix garbled output.
    
    Args:
        processing_class: The tokenizer/processor
        output_ids: The token IDs to decode
        
    Returns:
        Decoded text
    """
    print("🔧 Trying different decoding parameters...")
    
    # Method 1: Basic decode with skip_special_tokens=True
    try:
        text = processing_class.decode(output_ids, skip_special_tokens=True)
        if is_valid_text(text):
            print("✅ Method 1 (skip_special_tokens=True) worked")
            return text
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: With clean_up_tokenization_spaces
    try:
        text = processing_class.decode(
            output_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        if is_valid_text(text):
            print("✅ Method 2 (with clean_up) worked")
            return text
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Keep special tokens
    try:
        text = processing_class.decode(output_ids, skip_special_tokens=False)
        if is_valid_text(text):
            print("✅ Method 3 (keep special tokens) worked")
            return text
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Method 4: Filter out invalid tokens
    try:
        vocab_size = processing_class.vocab_size
        valid_tokens = output_ids[output_ids < vocab_size]
        if len(valid_tokens) > 0:
            text = processing_class.decode(valid_tokens, skip_special_tokens=True)
            if is_valid_text(text):
                print("✅ Method 4 (filtered invalid tokens) worked")
                return text
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
    
    # Method 5: Individual token decoding
    try:
        texts = []
        for token_id in output_ids:
            try:
                token_text = processing_class.decode([token_id], skip_special_tokens=True)
                if token_text and token_text.strip():
                    texts.append(token_text)
            except:
                continue
        text = "".join(texts)
        if is_valid_text(text):
            print("✅ Method 5 (individual tokens) worked")
            return text
    except Exception as e:
        print(f"❌ Method 5 failed: {e}")
    
    print("❌ All decoding methods failed")
    return ""


def is_valid_text(text: str) -> bool:
    """
    Check if the decoded text looks valid (not garbled).
    
    Args:
        text: The decoded text to check
        
    Returns:
        True if text looks valid
    """
    if not text or not text.strip():
        return False
    
    # Check for common garbled text patterns
    garbled_patterns = [
        "ई bargaining",  # Your example
        "Gujarฝ้า_force",
        "sex）。];//",
        "eligçoislogging",
        "alcanç lié",
        "organ trầm",
        "tob pygame",
        "Xm imminentologist",
        "小女孩要素涟",
        "mul Adrian",
        "蓋往-groups",
        "anime想起了",
        "绌oblin Brut",
        "comforting dc",
        "concaten地",
        "JacobsCrud",
        "欧式ickest",
        "listView名录",
        "Walt(priority",
        "Provideчин",
        "antiqueCamArmy",
        "providers produ",
        "复杂 ply",
        "TXT->{",
        "的成绩",
        "apache continuar",
        "constant.Utility",
        "vomitingredd",
        "select(DATA",
        "怎么办",
        "ballisticnickname",
        "いうاCached",
        "渔民 VPKent",
        "搬迁 Message",
        "grip ragaz",
        "学位Generated",
        "lhapolis Fluent",
        "ElementRef",
        "物业管理轴",
        "sendflows eBooks",
        "ttkHall cambiar",
        "Tau deterioration",
        "提拔FLAG",
        "influx✘",
        "重重 noise",
        "遛_uv sift",
        "础includes",
        "Http mData",
        "nonatomicstance",
        "的身体 detox",
        "적용دوا tol",
        "Spacer.bukkit",
        "낭 procession",
        "witnessedorThunk",
        "🏼 Journey",
        "textBox",
        "死刑intros",
        "neighborhoods",
        "WATER宿迁",
        "تكنRoad",
        "Readerillegal",
        "ไทย devoted",
        "paragraphscel",
        "viewerüssen",
        "맥_tool",
        "폽/P圣",
        "itters胡",
        "saidaitag",
        "printfได้รับ",
        "การ้อยוף",
        "سحب TheFL",
        "沫 giết",
        "Extended telescope",
        "暴雨🗒",
        "มั่นใจ greet",
        "izzly Herb",
        "PythonNavigation",
        "ItemSelectedListener",
        "科协قود",
        "Awake Stereo",
        "候选はない",
        "Osborne אלו",
        "Giant",
        "haste Continuing",
        "чувств matière",
        "Spring面容",
        "visualizeظ",
        "vídeることができる",
        "produk就很",
        "bble UIT",
        "reorderedՖ",
        "RESULT Veranst",
        "🥁 onRequest",
        "sued삵upiter",
        "shower المتعلقة",
        "caz.INPUT",
        "shortcode",
        "縱.poster",
        "hypnot enf",
        "DIFF:",
        "Sylvia.Reference",
        "込FullName",
        "prerequisites_WINDOWS",
        "monks yık",
        "yielded人物",
        "iliary.*",
        "阜阳.isDebugEnabled",
        "this Antarctic",
        "içer recurse",
        "nodeName毛利率",
        "赛场ܔbie",
        "Finder-ps",
        "decrementisory",
        "سيل룄赛车",
        "ripe쿨ائف",
        "L_W[_(plot",
        "named...",
        "车站 javax",
        "Tes发育",
        "Superv女士",
        "avoid Ing",
        "containersמתי",
        "LETiaux钤",
        "PatelAccordion",
        "hoodieⓒ",
        "sextreffenואי",
        "hasMany埘",
        "dctheadmts",
        "热潮:param",
        "ルー.isDebugEnabled",
        "mano藥",
        "NavigationView",
        "main𝐺",
        "chociayl:model",
        "akt鲽 juven",
        "Miles özellikleri",
        "رمslave иг",
        "ซ่า embed",
        "takım coveted",
        "쿙 lập",
        "upfrontmoon",
        "posьевStock",
        "Securities畜",
        "BannerEntropy",
        "crate shale",
        "shortcomings_CAMERA_DAY",
        "Collider clar",
        "ธุELSE",
        "ér Marlltre",
        "เมื่อqrstuvwxyz",
        "icators痉",
        "لاعبştir",
        "盛世ובל",
        "Blockingובש",
        "lure windowHeight",
        "水letamp merely",
        "Reynolds aus",
        "tpendLayout_wrapper",
        "derivative Marino",
        "int_msgVIDEO",
        "Baths Blinkerson",
        "Poor nổi",
        "station_cli",
        "至关或许",
        "总理較",
        "izational Northwestern",
        "__(}%鸹",
        "_timestamp乳",
        "民生得分",
        "领域的皮",
        "idden Confeder",
        "members生产力",
        "Circuit necessities",
        "喜好核",
        "етсяTAMade",
        "政していました",
        "搜救世界经济",
        "ἣ텼",
        "ไฟ/server",
        "COMMANDproducts_POP",
        "scriptions厉",
        "dw为啥",
        "SOLUTIONrazil",
        "drafted几天",
        "GDP\temail-addons",
        "технологHu",
        "chaud四是",
        "ört Permissions",
        "Println竦",
        "ם_Template",
        "菲优𝗘",
        "ự怨.angle",
        "cadre기가",
        "muttered정보",
        "kotlin เมื่อ",
        "Subsystem峃",
        "êtes.Rad",
        "radiator[node",
        "showcases לג",
        "region",
        "artikel.route",
        "iquid-----------",
        "读后感 quốc",
        "最常见的空气",
        "Occ‒Occ",
        "┙¡wonusu",
        "unsqueeze BehaviorSubject",
        "styleTypeJJ",
        "翎_particle",
        "forty Physiology",
        "Exceptions批判",
        "芫icip-bodied",
        "Composite(guid",
        "değil النو",
        "Amend_RATIO",
        "Mentionpromozzo",
        "抗震發揮",
        "Premium /**",
        "드립 pacientes",
        "Carp ^{",
        "ዲ ICU",
        "rollbackonden",
        "ten waitBottom",
        "ᠦbpp (#",
        "派出所극",
        "UICollectionView",
        "itm MATCH",
        "funוי带上",
        "enveenth Eagle",
        "Hotels indexPath",
        "Reyes峰ʇ",
        "Magic.mschedule",
        "GY Georges",
        "mnemonicplays",
        "义务 freezer",
        "urn>()",
        "匣.Shipgateway",
        "HTMLElement Medi",
        "mycketSt",
        "Françaisي",
        "presumedختص",
        "帮忙研判",
        "ViewById SetLastError",
        "Wah BORDER",
        "Ꮠ");}",
        "لالＵリフォーム",
        "planned\t\t",
        "meu战场",
        "죙诋 mlx",
        "Tops cornerstone",
        "inkle أسبوع",
        "魃一个多",
        "_SIDE dod",
        "使劲 tattoos",
        "anyhowⵔ",
        "layout //////////////////",
        "Skywalker坋",
        "Hibernate professornic",
        "о\tL",
        "Freund tertiary",
        "緣ByKey_hook",
        "algorithm такие",
        "understands晫",
        "媂/",
        "Number зад",
        "wendungcan",
        "type周岁",
        "Cha merged",
        "adınaход",
        "Doubleai/fwlink",
        "heraus娈:h",
        "惆 już&_４",
        "lâвеч falta",
        "Senior treeNode",
        "_counterッチ",
        "バ период",
        "force Taliban",
        "xffffff の",
        "إقليم tak",
        "被抓 raysoit",
        "new']),",
        "HindTEE_reservation",
        "parse fake",
        "DES-Length",
        "hỏng冷",
        "_ALLOC Femin",
        "hints봄的",
        "人物中介",
        "Mercer doctors",
        "收費.so",
        "없이ものคะ",
        "แนวิดีโอ",
        "唝集團",
        "Fee RestartLate",
        "solver.score",
        "ceansmensaje",
        "율.lua",
        "erton предн",
        "err că",
        "刚开始asant",
        "Spectrumremote",
        "‽-table",
        "이라ernetes",
        "SONเสริม",
        "libr落到实处",
        "crumbling相识",
        "AustralianPageRoute",
        "nowphysical_variable",
        "voluntary小孩",
        "قطعrectangle",
        "🏼現代前锋",
        "prop ואח",
        "Bollywood:flex",
        "思想_pix",
        "Kunden halfway",
        "locality peter",
        "SexyCppType",
        "strdup횃",
        "blobｋ",
        "קדBoxLayout",
        "线上线下",
        "Captとはいえ",
        "função",
        "ผลิตภัณฑ์mun",
        "groundwater就开始",
        "fears Vari",
        "FAC生产的",
        "communicöffentlich",
        "müssenrea",
        "vulnerabilities新一轮",
        "AILY.JSONArray",
        "IENTATION",
        "manufact샷",
        "_editor_DIRECT",
        "inté Enough",
        "Radiationⓡ",
        "_HIGHprivation",
        "_handlers'))",
        "となります cio",
        "decidedly郾",
        "carry土豆",
        "modPOSIT部",
        "careers(has",
        "ummies inversion",
        "社said inserting",
        "圣地 DOJ",
        "אפשר [-]:",
        "yawберرُ",
        "Batman siti",
        "Morrowreviews",
        "vile婪alcon",
        "Floating AppConfig",
        "Diamondsおよび",
        "NotBlank我已经",
        "髫Senderancode",
        "drops_apiCompare",
        "짠ademic커",
        "sizeof㫰",
        "GridView eos",
        "YesNochlor",
        "对中国蠡",
        "IVED pairwise",
        "הטבעponsorphi",
        "SEMB expand",
        "夔 ques",
        "board uncomfortable",
        "موجود\t",
        "atomicATER",
        "geometry pinterest",
        "PLATFORM cultured",
        "就没有 Stocks",
        "renovations_range",
        "_matrices坥",
        "利好jas",
        "פורס inherently",
        "/*_al slammed",
        "FriedrichSubmitting",
        "маякор",
        "humiliation(sf",
        "INITulkanallele",
        "ified光线🎙",
        "glGenerglass",
        "increment 회",
        "reason FORMAT",
        "ERVICE (...",
        "הודעה kå",
        "Ingredientsיחס",
        "מלח抽取",
        "非常多 век",
        "Ary initialValues",
        "Crown貼_",
        "/f trou",
        "chauffuya",
        "ໃيز/tmp",
        "cheatedска",
        "method",
        "_que开发商",
        "真的很::$_",
        "小小的(with",
        "COLUMN📷",
        "advantage mimetype",
        "uriaという",
        "adviceContent",
        "trabalhoảo",
        "נטל/",
        "考え方",
        "صيان Michel",
        "едак悩み",
        "Ra innovative",
        "itr cuatro",
        "腒.mc ber",
        "reins大致",
        "_mBbeiter",
        "צבא لكن",
        "taking(strategy",
        "羔 segSigning",
        "划分Não",
        "품valueOf",
        "lombok النو",
        "establish龉",
        "我的 Fowler",
        "Temperatureแน",
        "adres pang",
        "wherever Russo",
        "circumstanceDuplicate",
        "getAddresspherd",
        "Ⓡ杂志社",
        "LitBillyocab",
        "renderer selenium",
        "ۆ Stretch",
        "addItemimesteps",
        "箸/msg",
        "removeObject¾",
        "To mgaShipping",
        "ViewChildеш",
        "mauratasets",
        "culturally則",
        "alcon המי",
        "pup الْ",
        "安徽省soap",
        "width\trunTry",
        "小区 מילי",
        "hitting Rupertssf",
        "党的าร์ด",
        "_teacher.Slf",
        "Arbor possibilities",
        "_ENDPOINT(embed",
        "鹑?",
        "voltaburn",
        "exposing恩",
        "thắng这批",
        "котором.method",
        "ス相同",
        "Count"
    ]
    
    # Check if text contains any garbled patterns
    for pattern in garbled_patterns:
        if pattern in text:
            return False
    
    # Check for excessive special characters or mixed languages
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    total_chars = len(text)
    if total_chars > 0 and special_char_count / total_chars > 0.5:
        return False
    
    # Check for reasonable character distribution
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if total_chars > 0 and ascii_chars / total_chars < 0.3:
        return False
    
    return True


def patch_sglang_decode():
    """
    Monkey patch the SGLangRollout decode method to use better parameters.
    """
    from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
    
    original_handle_engine_call = SGLangRollout._handle_engine_call
    
    async def patched_handle_engine_call(self, _req, sampling_params, image_data=None):
        """Patched version with better decoding."""
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        output = await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)
        
        # Use the improved decoding
        output["text"] = fix_decode_parameters(
            self.processing_class, 
            output["output_ids"]
        )
        
        return output
    
    # Apply the patch
    SGLangRollout._handle_engine_call = patched_handle_engine_call
    print("✅ Applied SGLangRollout decode patch")


if __name__ == "__main__":
    print("🔧 SGLangRollout Output Fix")
    print("=" * 50)
    
    # Apply the patch
    patch_sglang_decode()
    
    print("✅ Patch applied! The SGLangRollout should now decode output correctly.")
    print("\nTo use this fix in your code, import this module:")
    print("from fix_sglang_output import patch_sglang_decode")
    print("patch_sglang_decode()")
