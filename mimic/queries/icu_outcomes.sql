SELECT
  icustays.*,
  gi_patients.gender,
  gi_patients.anchor_age,
  gi_patients.anchor_year,
  gi_patients.anchor_year_group,
  gi_patients.insurance,
  gi_patients.language,
  gi_patients.marital_status,
  gi_patients.race,
  admissions.deathtime,
  admissions.deathtime BETWEEN icustays.intime AND icustays.outtime AS icu_expire_flag,
  admissions.hospital_expire_flag
FROM
  `physionet-data.mimiciv_icu.icustays` AS icustays
JOIN
  `mimic-427018.gi_bleeding.gi_bleeding_patients` AS gi_patients
ON
  icustays.hadm_id = gi_patients.hadm_id
JOIN
  `physionet-data.mimiciv_hosp.admissions` AS admissions
  ON icustays.hadm_id = admissions.hadm_id
