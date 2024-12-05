-- DEPENDS ON `gi_patients.sql`
-- All blood transfusions for patients with a GI-Bleeding diagnosis

SELECT
  d_items.label AS item_label,
  inputevents.stay_id,
  inputevents.starttime,
  inputevents.endtime,
  inputevents.itemid,
  inputevents.amount,
  inputevents.rate,
  inputevents.isopenbag,
  inputevents.totalamount,
  gi_bleeding_patients.*
FROM
  physionet-data.mimiciv_icu.inputevents
JOIN
  mimic-427018.gi_bleeding.gi_bleeding_patients
ON
  inputevents.hadm_id = gi_bleeding_patients.hadm_id
JOIN
  physionet-data.mimiciv_icu.d_items
ON
  inputevents.itemid = d_items.itemid
WHERE
  inputevents.itemid IN (
    225168, -- Packed Red Blood Cells
    226370, -- OR Autologous Blood Intake
    221013, -- Whole Blood
    226368, -- OR Packed RBC Intake
    227070  -- PACU Packed RBC Intake
  )
ORDER BY
  inputevents.stay_id,
  inputevents.endtime ASC
