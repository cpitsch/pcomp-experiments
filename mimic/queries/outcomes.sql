-- Get the admission and outcome for each patient
SELECT
  admissions.subject_id,
  admissions.hadm_id,
  icu_stays.stay_id,
  admissions.admittime,
  admissions.dischtime,
  admissions.hospital_expire_flag > 0 AS hospital_expire_flag
  gi_bleeding_patients.*
FROM
  `physionet-data.mimiciv_hosp.admissions` AS admissions
JOIN
  `mimic-427018.gi_bleeding.gi_bleeding_patients` AS gi_bleeding_patients
ON
  gi_bleeding_patients.hadm_id = admissions.hadm_id
JOIN
  `physionet-data.mimiciv_icu.icustays` AS icu_stays
ON
  icu_stays.hadm_id = admissions.hadm_id
