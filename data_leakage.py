def check_for_leakage(df1, df2, patient_col):
    df1_patients_unique = set(df1[patient_col].unique())
    df2_patients_unique = set(df2[patient_col].unique())

    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)
    leakage = len(patients_in_both_groups) > 0

    return leakage