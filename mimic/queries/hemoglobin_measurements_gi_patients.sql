-- DEPENDS ON `gi_patients.sql`
-- Columns: stay_id, starttime, endtime, itemid, amount, rate, isopenbag, totalamount, [gi_patients.*, hadm_id, ...]

SELECT
  icustays.stay_id,
  d_labitems.label,
  labevents.labevent_id,
  labevents.charttime,
  labevents.storetime,
  labevents.valuenum,
  labevents.ref_range_lower,
  labevents.ref_range_upper,
  gi_bleeding_patients.*
FROM
  `physionet-data.mimiciv_hosp.labevents` AS labevents
JOIN
  `physionet-data.mimiciv_hosp.d_labitems` AS d_labitems
ON
  labevents.itemid = d_labitems.itemid
JOIN
  `mimic-427018.gi_bleeding.gi_bleeding_patients` AS gi_bleeding_patients
ON
  labevents.hadm_id = gi_bleeding_patients.hadm_id
JOIN
  `physionet-data.mimiciv_icu.icustays` AS icustays
ON
  -- Only associate those hemoglobin measurements to an ICU stay which were charted during the stay (TODO: or also within some window beforehand?)
  icustays.hadm_id = labevents.hadm_id AND labevents.charttime BETWEEN icustays.intime AND icustays.outtime
WHERE
    labevents.itemid IN (
      51640, 50811, 51222 -- Hemoglobin
      -- 51645  -- Hemoglobin, calculated
      )
ORDER BY
    labevents.hadm_id,
    labevents.charttime ASC
