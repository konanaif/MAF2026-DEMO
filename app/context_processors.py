ALGORITHM_NAME2ID = {
    # tabular
    # preprocessing
    "DisparateImpactRemover": "dir",
    "LearningFairRepresentation": "lfr",
    "Reweighing": "rw",
    "OptimPreproc": "optimpreproc",
    "FairStreamingPCA": "fairpca",
    "FairBatch": "fairbatch",
    # inprocessing
    "MetaFairClassifier": "mfc",
    "PrejudiceRemover": "prejremover",
    "SLIDE": "slide",
    "FTM": "ftm",
    "ExponentiatedGradientReduction": "egr",
    "KernelDensityEstimation": "kde",
    "GerryFairClassifier": "gfc",
    "GridSearchReduction": "gsr",
    "AdversarialDebiasing": "ad",
    "sIPMLFR": "sipmlfr",
    "ALINCredit": "alincredit",
    "DRAF": "draf",
    "DEEMC": "deemc",
    # postprocessing
    "CalibratedEqOdds": "ceo",
    "EqualizedOdds": "eo",
    "RejectOptionClassifier": "roc",
    "CasualPathTracing": "cpt",
    "Ember": "ember",
    # image-inprocessing
    "FairDimensionFiltering": "fdf",
    "FairFeatureDistillation": "ffd",
    "FairnessVAE": "fvae",
    "LearningFromFairness": "lff",
    "DMLBG": "dmlbg",
    "SubgroupBalancedResampling": "sbr",
    # audio
    "INTapt": "intapt",
    "FairASR": "fairasr",
    # text
    "RepresentativenessHeuristic": "rh",
    "Latte": "latte",
    "CREHate": "crehate",
    "KoBBQ": "kobbq",
    "BBG": "bbg",
    "ConCSE": "concse",
    "KoBBQSteer": "kobbqsteer",
}

ALGORITHM_ID2NAME = {
    ALGORITHM_NAME2ID[algo_id]: algo_id for algo_id in ALGORITHM_NAME2ID.keys()
}


def algorithm_id(request):
    return ALGORITHM_NAME2ID
