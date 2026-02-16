#! /usr/bin/env python

#Specimen,Primary Target,Flavor Profile,"Note on the ""Mother"""
#Mugwort / Wormwood,Thujone (GABA-A),Intense Bitter / Sage,"Use sparingly; the backbone of the ""kick."""
#Yarrow,Achilleine / Thujone,Medicinal / Astringent,"Adds a ""wild"" floral note; very traditional."
#Damiana,Dopamine / Libido,Sweet / Woody / Cacao,Excellent for masking the bitterness of wormwood.
#Bog Myrtle,Essential Oils (Lucid),Resinous / Eucalyptus,"Provides the ""heady"" aroma of ancient gruit."
#California Poppy,GABA / Opioid (Mild),Earthy / Bitter,"Very stable; gives the ""body"" a heavy, relaxed feel."
#Passionflower,GABA / Mild MAOI,Grassy / Neutral,"The ""potentiator""—makes other herbs feel stronger."
#Lemon Balm,GABA-T Inhibitor,Bright Citrus / Mint,"Essential to lift the heavy, ""muddy"" root flavors."
#Rhodiola Rosea,Dopamine / Serotonin,Rose-like / Tannic,"Adds structure (tannins) and ""sustain"" to the mood."
#Mucuna Pruriens,L-Dopa (Dopamine),Nutty / Bland,"Best added as a powder to the final ""juice"" blend."
#Sceletium (Kanna),Serotonin (SRI),Sour / Succulent,"The ""Euphoric."" Improves with the mother’s acidity."
#Albizia (Mimosa),Serotonin / Spirit,Sweet / Tannic,"The bark adds a deep ""cured"" wood flavor."
#Gotu Kola,GABA / Neurogenic,Mild / Green,"Pairs perfectly with the Matcha ""green"" profile."
#Skullcap,GABA-A,"Bitter / ""Thin""","The ""Chill Switch."" Powerful for relaxation."
#Black/Green Tea,Caffeine / L-Theanine,Tannic / Umami,"Provides the ""Structure"" and the energy base."
#Ginger / Turmeric,Circulation (Vaso),Spicy / Earthy,"The ""Engine."" Keeps the blood flowing and the gut happy."
#
#Pairing Type,Specimen A,Specimen B,Result
#"The ""Limitless"" Stack",Matcha / Green Tea,Gotu Kola,Sharp focus without the caffeine jitters.
#"The ""Deep Sleep""",California Poppy,Skullcap,Highly sedative; save for evening draws.
#"The ""Heart Opener""",Kanna (Sceletium),Damiana,"High euphoria and ""social"" energy."
#"The ""Vasodilator""",Ginger / Turmeric,Black Tea / LSA*,"Offsets the ""clamping"" of caffeine or seeds."
#
#Kanna + Passionflower: Both affect serotonin/MAO. Use small amounts of each if mixing, as they can "stack" unpredictably.
#
#Wormwood + Skullcap: Both hit GABA-A but in different ways (one is an antagonist, one is an agonist). Mixing large amounts can cause "neurological static" or confusion.
#
#St. John's Wort (Reminder): Still excluded because it "fights" with almost everything on this list via liver enzyme induction.

from enum     import Enum
from types    import *
from typing   import *

from pydantic import BaseModel

class Neurotransmitter(Enum): # TODO comprehensive list
    SEROTONIN      = 'serotonin'
    NOREPINEPHRINE = 'norepinephrine'
    ACETYLCHOLINE  = 'acetylcholine'
    GLUTAMATE      = 'glutamate'
    GABA_A         = 'gaba_a'
    GABA_T         = 'gaba_t'
    ENDORPHINS     = 'endorphins'
    DOPAMINE       = 'dopamine'
    CANNABANOIDS   = 'cannabanoids'
    CORTISOL       = 'cortisol'
    ADRENALINE     = 'adrenaline'
    THYROXINE      = 'thyroxine'
    TESTOSTERONE   = 'testosterone'
    DHEA           = 'dhea'
    OXYTOCIN       = 'oxytocin'
    PROGESTERONE   = 'progesterone'
    ESTROGEN       = 'estrogen'

    # TODO needs review
    ADENOSINE      = 'adenosine'
    GLUTAMATE      = 'glutamate'

    # TODO VASCULAR ??? i.e., with mechanism:
    # - VASODILATOR
    # - VASOCONSTRICTOR

class Mechanism(Enum): # TODO comprehensive list
    AGONIST    = 'agonist'
    ANTAGONIST = 'antagonist'
    INHIBITOR  = 'inhibitor'
    SRI        = 'sri'
    PRECURSOR  = 'precursor'
    MAOI_A     = 'maoi_a'
    MAOI_B     = 'maoi_b'
    MODULATOR  = 'modulator'

nt_mech_to_nt: Dict[Tuple[Neurotransmitter, Mechanism], Neurotransmitter] = {
    (Neurotransmitter.SEROTONIN, Mechanism.SRI): Neurotransmitter.SEROTONIN,
    (Neurotransmitter.DOPAMINE, Mechanism.MAOI_B): Neurotransmitter.DOPAMINE,
    (Neurotransmitter.GABA_A, Mechanism.AGONIST): Neurotransmitter.GABA_A,
    (Neurotransmitter.GABA_A, Mechanism.ANTAGONIST): Neurotransmitter.GABA_A,
    (Neurotransmitter.ADENOSINE, Mechanism.ANTAGONIST): Neurotransmitter.ADRENALINE, # Blocking adenosine triggers adrenaline
}

Plant                         :TypeAlias           = str
plants                        :List[Plant]         = [
    "mugwort", "wormwood", "yarrow", "damiana", "bog_myrtle", 
    "california_poppy", "passion_flower", "lemon_balm", "rhodiola_rosea", 
    "mucuna_pruriens", "sceletium_tortuosum", "albizia_jubrissin", 
    "gotu_kola", "skullcap", "black_tea", "matcha", "ginger", "turmeric"
]

# TODO plant parts! (different parts ==> different flavor, different effect)

nt_mech_to_plant: Dict[Tuple[Neurotransmitter, Mechanism], Set[Plant]] = {
    (Neurotransmitter.SEROTONIN, Mechanism.SRI): {"sceletium_tortuosum"},
    (Neurotransmitter.DOPAMINE, Mechanism.MAOI_B): {"rhodiola_rosea", "passion_flower"},
    (Neurotransmitter.GABA_A, Mechanism.ANTAGONIST): {"wormwood", "mugwort"},
    (Neurotransmitter.GABA_A, Mechanism.AGONIST): {"skullcap", "california_poppy"},
}

class Flavor(Enum):
    BITTER     = 'bitter'
    SWEET      = 'sweet'
    SOUR       = 'sour'
    SALTY      = 'salty'
    UMAMI      = 'umami'

    # needs review
    WOODY      = 'woody'
    RESINOUS   = 'resinous'
    CAMPHOR    = 'camphor'
    CITRUS     = 'citrus'
    EARTHY     = 'earthy'
    NUTTY      = 'nutty'
    TANNIC     = 'tannic'
    ASTRINGENT = 'astringent'
    FUNKY      = 'funky'
    BRIGHT     = 'bright'
    MINERAL    = 'mineral'

class FlavorMechanism(Enum):
    MASK      = 'mask'       # Stronger flavor hides a weaker/unpleasant one
    BRIDGE    = 'bridge'     # Connects two disparate flavors (e.g., Earthy to Bright)
    AMPLIFY   = 'amplify'    # Enhances a specific note
    CLASH     = 'clash'      # Chemically or sensorially unpleasant together
    BALANCE   = 'balance'    # Neutralizes a harsh quality (like Tannin vs Sour)

# (Flavor, FlavorMechanism) -> Target_Flavor
flavor_mech_to_flavor: Dict[Tuple[Flavor, FlavorMechanism], Flavor] = {
    (Flavor.SWEET, FlavorMechanism.MASK): Flavor.BITTER,
    (Flavor.WOODY, FlavorMechanism.BRIDGE): Flavor.BITTER, # Woody/Damiana bridges bitter to sweet
    (Flavor.CITRUS, FlavorMechanism.BRIGHT): Flavor.EARTHY, # Brightens heavy roots
    (Flavor.TANNIC, FlavorMechanism.BALANCE): Flavor.SOUR,
    (Flavor.BITTER, FlavorMechanism.CLASH): Flavor.RESINOUS, # Harsh + Harsh = Bad
}

flavor_table                  :Dict[Flavor,Set[Plant]] = {
        Flavor.BITTER    : {"mugwort", "wormwood", "yarrow", "skullcap", "california_poppy"},
        Flavor.SWEET     : {"damiana", "albizia_jubrissin"},
        Flavor.WOODY     : {"damiana", "albizia_jubrissin"},
        Flavor.RESINOUS  : {"bog_myrtle", "yarrow"},
        Flavor.CAMPHOR   : {"bog_myrtle", "yarrow"},
        Flavor.CITRUS    : {"lemon_balm", "ginger"},
        Flavor.BRIGHT    : {"lemon_balm", "ginger"},
        Flavor.EARTHY    : {"mucuna_pruriens", "turmeric", "gotu_kola"},
        Flavor.NUTTY     : {"mucuna_pruriens", "turmeric", "gotu_kola"},
        Flavor.TANNIC    : {"black_tea", "matcha", "rhodiola_rosea", "albizia_jubrissin"},
        Flavor.ASTRINGENT: {"black_tea", "matcha", "rhodiola_rosea", "albizia_jubrissin"},
        Flavor.SOUR      : {"sceletium_tortuosum"},
        Flavor.FUNKY     : {"sceletium_tortuosum"},
}

class CompoundCategory(Enum):
    ALKALOID     = 'alkaloid'     # Needs Acid (Vat of Acid)
    TERPENE      = 'terpene'      # Needs Heat or Alcohol (Aromatic Wort)
    GLYCOSIDE    = 'glycoside'    # Needs Time/Fermentation (Trash Bucket)
    TANNIN       = 'tannin'       # Extractable in Water/Acid (Acid or Diluent)
    MUCILAGE     = 'mucilage'     # Cold extractable (Diluent)
    VOLATILE_OIL = 'volatile_oil' # Delicate; no heat (Bottling/Trash Bucket)

extraction_logic: Dict[CompoundCategory, str] = {
    CompoundCategory.ALKALOID:     "vat_of_acid",
    CompoundCategory.TERPENE:      "aromatic_wort",
    CompoundCategory.GLYCOSIDE:    "trash_bucket",
    CompoundCategory.VOLATILE_OIL: "bottling_mod",
    CompoundCategory.TANNIN:       "aromatic_wort", # Heat pulls heavy tannins
}

plant_to_compounds: Dict[Plant, Set[CompoundCategory]] = {
    "sceletium_tortuosum": {CompoundCategory.ALKALOID},
    "mugwort":             {CompoundCategory.TERPENE, CompoundCategory.VOLATILE_OIL},
    "wormwood":            {CompoundCategory.TERPENE},
    "lemon_balm":          {CompoundCategory.VOLATILE_OIL},
    "california_poppy":    {CompoundCategory.ALKALOID},
    "black_tea":           {CompoundCategory.TANNIN},
    "rhodiola_rosea":      {CompoundCategory.ALKALOID, CompoundCategory.TANNIN},
    "chamomile":           {CompoundCategory.MUCILAGE, CompoundCategory.VOLATILE_OIL},
}

class Extraction(Enum):
    RAW             = 'raw'          # no special processing -- just throw it into the mother
    DECOCT          = 'decoct'       # Boil/Reduce solids BEFORE adding liquid to Mother
    STEEP           = 'steep'        # not used
    INFUSE          = 'infuse'       # cold infusion; delicate; added at bottling
    EXTRACT         = 'extract'      # oil; not used
    BUCKET_ACID     = 'bucket_acid'  # for alkaloid salts
    BUCKET_SUGAR    = 'bucket_sugar' # for oleo saccarum
    EMULSIFY        = 'emulsify'     # for oils
    BLEND           = 'blend'        # use the blender (wet)
    GRIND           = 'grind'        # use the coffee grinder (dry); for making oils
    PRESS           = 'press'        # juice it
    POWDER          = 'powder'       # the matcha
    REDUCE          = 'reduce'

compound_to_extraction:Dict[CompoundCategory,Set[Extraction]] = {

}

class Stage(Enum):
    BUCKET_SUGAR = 'bucket_sugar'
    BUCKET_ACID  = 'bucket_acid'
    WORT         = 'wort'
    MOTHER       = 'mother'
    BUCKET_WINE  = 'bucket'
    CARBUOY      = 'carbuoy'
    RACKING      = 'racking'
    BOTTLING     = 'bottling'
    BATCHING     = 'batching'
    SERVING      = 'serving'

# TODO stage to timing table

# TODO stage to extraction list
stage_to_extraction:Dict[Stage,List[Extraction]] = {
        # FIXME this doesn't work. it fails to represent that the ginger/turmeric are blended raw and pressed, then the fresh juice is added to the mother, then the blended, pressed ginger/turmeric is decocted, pressed, reduced till thick, then added to the mother
    Stage.WORT: [Extraction.BLEND, Extraction.PRESS, Extraction.DECOCT, Extraction.PRESS, Extraction.REDUCE]
}

class SpecimenInstruction(BaseModel):
    plant: Plant
    primary_stage: Stage       # Where the bulk of the biomass goes
    extraction_method: Extraction
    physical_action: List[str] # ["Blend", "Decoct Solids", "Discard Solids"]
    bottling_note: Optional[str] = None
    
# Mapping the "Hard" and "Soft" logic into the compiler
potion_manifest: Dict[Plant, SpecimenInstruction] = {
    "ginger": SpecimenInstruction(
        plant="ginger",
        primary_stage="aromatic_wort",
        extraction_method=Extraction.DECOCT,
        physical_action=["Blend Raw", "Squeeze", "Boil Juice/Solids", "Reduce 50%"]
    ),
    "mugwort": SpecimenInstruction(
        plant="mugwort",
        primary_stage="trash_bucket_early",
        extraction_method=Extraction.BUCKET_SUGAR,
        physical_action=["Bruise", "Ferment in Oleo", "Squeeze into Mother"]
    ),
    "sceletium_tortuosum": SpecimenInstruction(
        plant="sceletium_tortuosum",
        primary_stage="vat_of_acid",
        extraction_method=Extraction.BUCKET_ACID,
        physical_action=["Crush", "Citrus Soak", "Add Liquid to Mother"]
    ),
}


# stages:
# 1a) sugar bucket
# 1b)acid bucket
# 1c) decoction & reduction (ginger, turmeric, tree bark, caramel... basically water-soluble things that I can concentrate)
# 2) mother
# 3) carbuoy
# 4) bottling
# 5) batching/serving

process_stages = {
    "trash_bucket_early": {
        "timing": "Days 1-7",
        "goal": "Ferment fruit/sugar",
        "compounds": [CompoundCategory.GLYCOSIDE]
    },
    "trash_bucket_late": {
        "timing": "Days 5-7",
        "goal": "Delicate aromatics & tea (prevents over-steeping)",
        "compounds": [CompoundCategory.VOLATILE_OIL]
    },
    "aromatic_wort": {
        "timing": "Feeding Day",
        "goal": "Heavy extraction (Decoction)",
        "compounds": [CompoundCategory.TERPENE, CompoundCategory.TANNIN]
    },
    "vat_of_acid": {
        "timing": "Days 1-3",
        "goal": "Salt conversion",
        "compounds": [CompoundCategory.ALKALOID]
    }
}


