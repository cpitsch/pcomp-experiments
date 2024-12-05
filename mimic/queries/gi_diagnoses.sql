-- Contains all GI-Bleeding Diagnoses
-- Columns: subject_id, hadm_id, icd_code, long_title

CREATE SCHEMA IF NOT EXISTS `mimic-427018.gi_bleeding`
OPTIONS (
  location = "US"
);

CREATE OR REPLACE TABLE `mimic-427018.gi_bleeding.gi_bleeding_diagnoses` AS (
  SELECT
    diagnoses_icd.*,
    d_icd_diagnoses.long_title
  FROM
    physionet-data.mimiciv_hosp.diagnoses_icd
  JOIN
    physionet-data.mimiciv_hosp.d_icd_diagnoses
  ON
    d_icd_diagnoses.icd_code = diagnoses_icd.icd_code
  WHERE (
      diagnoses_icd.icd_version = 9 AND
      diagnoses_icd.icd_code IN (
        "5780", -- Hematemesis
        "5307", -- Gastroesophageal Laceration-Hemorrhage Syndrome
        "53021", -- Ulcer Of Esophagus With Bleeding
        "53082", -- Esophageal Hemorrhage
        "4560", -- Esophageal Varices With Bleeding
        "4562", -- Secondary Esophageal Varices With Bleeding
        "5310", -- Acute Gastric Ulcer With Hemorrhage
        "5310", -- Acute Gastric Ulcer With Hemorrhage, Without Mention Of Obstruction
        "53101", -- Acute Gastric Ulcer With Hemorrhage, With Obstruction
        "5312", -- Acute Gastric Ulcer With Hemorrhage And Perforation
        "5312", -- Acute Gastric Ulcer With Hemorrhage And Perforation, Withoutmention Of Obstruction
        "53121", -- Acute Gastric Ulcer With Hemorrhage And Perforation, With Obstruction
        "5314", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage
        "5314", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage, Withoutmention Of Obstruction
        "53141", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage, With Obstruction
        "5316", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage Andperforation
        "5316", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage And Perforation, Withoutmention Of Obstruction
        "53161", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage And Perforation, With Obstruction
        "5320", -- Acute Duodenal Ulcer With Hemorrhage
        "5320", -- Acute Duodenal Ulcer With Hemorrhage, Withoutmention Of Obstruction
        "53201", -- Acute Duodenal Ulcer With Hemorrhage, With Obstruction
        "5322", -- Acute Duodenal Ulcer With Hemorrhage And Perforation
        "5322", -- Acute Duodenal Ulcer With Hemorrhage And Perforation, Withoutmention Of Obstruction
        "53221", -- Acute Duodenal Ulcer With Hemorrhage And Perforation, With Obstruction
        "5324", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage
        "5324", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage, Withoutmention Of Obstruction
        "53241", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage, Withobstruction
        "5326", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage And Perforation
        "5326", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage And Perforation, Withoutmention Of Obstruction
        "53261", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage And Perforation, Withobstruction
        "5330", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage
        "5330", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage, Withoutmention Of Obstruction
        "53301", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage, With Obstruction
        "5332", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation
        "5332", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation, Withoutmention Of Obstruction
        "53321", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation, With Obstruction
        "5334", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage
        "5334", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage, Without Mention Of Obstruction
        "53341", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage, With Obstruction
        "5336", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation
        "5336", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation, Without Mention Of Obstruction
        "53361", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation, With Obstruction
        "5340", -- Acute Gastrojejunal Ulcer With Hemorrhage
        "5340", -- Acute Gastrojejunal Ulcer With Hemorrhage, Without Mention Of Obstruction
        "53401", -- Acute Gastrojejunal Ulcer With Hemorrhage, With Obstruction
        "5342", -- Acute Gastrojejunal Ulcer With Hemorrhage And Perforation
        "5342", -- Acute Gastrojejunal Ulcer With Hemorrhage And Perforation, Without Mention Of Obstruction
        "53421", -- Acute Gastrojejunal Ulcer With Hemorrhage And Perforation, With Obstruction
        "5344", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage
        "5344", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage, Without Mention Of Obstruction
        "53441", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage, With Obstruction
        "5346", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage And Perforation
        "5346", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage And Perforation, Without Mention Of Obstruction
        "53461", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage And Perforation, With Obstruction
        "53501", -- Acute Gastritis, With Hemorrhage
        "53511", -- Atrophic Gastritis, With Hemorrhage
        "53521", -- Gastric Mucosal Hypertrophy, With Hemorrhage
        "53531", -- Alcoholic Gastritis, With Hemorrhage
        "53541", -- Other Specified Gastritis, With Hemorrhage
        "53551", -- Unspecified Gastritis And Gastroduodenitis, With Hemorrhage
        "53561", -- Duodenitis, With Hemorrhage
        "53571", -- Eosinophilic Gastritis, With Hemorrhage
        "53784", -- Dieulafoy Lesion (Hemorrhagic) Of Stomach And Duodenum
        "53783"  -- Angiodysplasia Of Stomach And Duodenum With Hemorrhage
      )
  ) OR (
    diagnoses_icd.icd_version = 10 AND
    diagnoses_icd.icd_code IN (
      "K920", -- Hematemesis
      "K226", -- Gastroesophageal Laceration-Hemorrhage Syndrome
      "K2211", -- Ulcer Of Esophagus With Bleeding
      "I8501", -- Esophageal Varices With Bleeding
      "I8511", -- Secondary Esophageal Varices With Bleeding
      "K250", -- Acute Gastric Ulcer With Hemorrhage
      "K252", -- Acute Gastric Ulcer With Hemorrhage And Perforation
      "K254", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage
      "K256", -- Chronic Or Unspecified Gastric Ulcer With Hemorrhage Andperforation
      "K260", -- Acute Duodenal Ulcer With Hemorrhage
      "K262", -- Acute Duodenal Ulcer With Hemorrhage And Perforation
      "K264", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage
      "K266", -- Chronic Or Unspecified Duodenal Ulcer With Hemorrhage And Perforation
      "K270", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage
      "K272", -- Acute Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation
      "K274", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage
      "K276", -- Chronic Or Unspecified Peptic Ulcer Of Unspecified Site With Hemorrhage And Perforation
      "K280", -- Acute Gastrojejunal Ulcer With Hemorrhage
      "K282", -- Acute Gastrojejunal Ulcer With Hemorrhage And Perforation
      "K284", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage
      "K286", -- Chronic Or Unspecified Gastrojejunal Ulcer With Hemorrhage And Perforation
      "K2901", -- Acute Gastritis, With Hemorrhage
      "K2941", -- Atrophic Gastritis, With Hemorrhage
      "K2961", -- Gastric Mucosal Hypertrophy, With Hemorrhage
      "K2921", -- Alcoholic Gastritis, With Hemorrhage
      "K2951", --
      "K2961", -- Other Specified Gastritis, With Hemorrhage
      "K2971", -- Unspecified Gastritis And Gastroduodenitis, With Hemorrhage
      "K2991", --
      "K2981", -- Duodenitis, With Hemorrhage
      "K3182", -- Dieulafoy Lesion (Hemorrhagic) Of Stomach And Duodenum
      "K2081", -- Other Esophagitis With Bleeding
      "K2091", -- Esophagitis Unspecified With Bleeding
      "K2101", -- Gastroesophageal Reflux Disease With Esophagitis With Bleeding
      "K31811"  -- Angiodysplasia Of Stomach And Duodenum With Hemorrhage
    )
  )
)
