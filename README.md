# msoft20
dataset: sha256 hashes of the Android apps used in MobileSoft20 paper
    benign_sha256.csv: hashes of the benign samples
    malware_sha256.csv: hashes of the malware samples

csv_files: features extracted from dataset
    smsf.csv.gz: static-sequence features
    scuf.csv.gz: static-use features
    dmsf.csv.gz: dynamic-sequence features
    dcuf.csv.gz: dynamic-use features
    hmsf.csv.gz: hybrid-sequence features
    hcuf.csv.gz: hybrid-use features
    Download at https://smu.sg/iwm

scripts: Python scripts for machine learning and deep learning classifiers
    ml.py: Python script for running machine learning classifiers on the csv_files
    dl.py: Python script for running deep learning classifiers on the csv_files
    msoft_env.yml: Python dependencies