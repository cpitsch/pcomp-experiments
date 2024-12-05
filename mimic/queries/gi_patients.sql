-- DEPENDS ON `gi_diagnoses.sql`
-- (Demographic) Information for all GI-Bleeding Patients
-- Columns: subject_id, hadm_id, gender, anchor_age (...), insurance, language, marital_status, race

CREATE SCHEMA IF NOT EXISTS `mimic-427018.gi_bleeding`
OPTIONS (
  location = "US"
);

CREATE OR REPLACE TABLE `mimic-427018.gi_bleeding.gi_bleeding_patients` AS 
WITH gi_patients AS (
  SELECT
    gi_bleeding_diagnoses.subject_id,
    gi_bleeding_diagnoses.hadm_id,
    patients.gender,
    patients.anchor_age,
    patients.anchor_year,
    patients.anchor_year_group,
    admissions.insurance,
    admissions.language,
    admissions.marital_status,
    admissions.race
  FROM
    mimic-427018.gi_bleeding.gi_bleeding_diagnoses
  JOIN physionet-data.mimiciv_hosp.patients ON gi_bleeding_diagnoses.subject_id = patients.subject_id
  JOIN physionet-data.mimiciv_hosp.admissions ON gi_bleeding_diagnoses.hadm_id = admissions.hadm_id
)
SELECT DISTINCT * FROM gi_patients WHERE anchor_age >= 18;
