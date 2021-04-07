import pandas as pd
from config import DATASET_DIR

""" 
Open University Learning Analytics Dataset (OULAD)
This dataset contains data about courses, students and their interactions with Virtual Learning Environment (VLE) for seven selected courses (called modules). Presentations of courses start in February and October - they are marked by “B” and “J” respectively. The dataset consists of tables connected using unique identifiers. All tables are stored in the csv format. 
│
└─ courses.csv
│  └─ File contains the list of all available modules and their presentations. The columns are:
│  │  └─ code_module – code name of the module, which serves as the identifier.
│  │  └─ code_presentation – code name of the presentation. It consists of the year and “B” for the presentation starting in February and “J” for the presentation starting in October.
│  │  └─ length - length of the module-presentation in days.
│  └─ The structure of B and J presentations may differ and therefore it is good practice to analyse the B and J presentations separately. Nevertheless, for some presentations the corresponding previous B/J presentation do not exist and therefore the J presentation must be used to inform the B presentation or vice versa. In the dataset this is the case of CCC, EEE and GGG modules.
│
└─ assessments.csv
│  └─ This file contains information about assessments in module-presentations. Usually, every presentation has a number of assessments followed by the final exam. CSV contains columns:
│  │  └─ code_module – identification code of the module, to which the assessment belongs.
│  │  └─ code_presentation - identification code of the presentation, to which the assessment belongs.
│  │  └─ id_assessment – identification number of the assessment.
│  │  └─ assessment_type – type of assessment. Three types of assessments exist: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA) and Final Exam (Exam).
│  │  └─ date – information about the final submission date of the assessment calculated as the number of days since the start of the module-presentation. The starting date of the presentation has number 0 (zero).
│  │  └─ weight - weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%.
│  └─ If the information about the final exam date is missing, it is at the end of the last presentation week.
│
└─ vle.csv
│  └─ The csv file contains information about the available materials in the VLE. Typically these are html pages, pdf files, etc. Students have access to these materials online and their interactions with the materials are recorded. The vle.csv file contains the following columns:
│     └─ id_site – an identification number of the material.
│     └─ code_module – an identification code for module.
│     └─ code_presentation - the identification code of presentation.
│     └─ activity_type – the role associated with the module material.
│     └─ week_from – the week from which the material is planned to be used.
│     └─ week_to – week until which the material is planned to be used.
│
└─ studentInfo.csv
│  └─ This file contains demographic information about the students together with their results. File contains the following columns:
│     └─ code_module – an identification code for a module on which the student is registered.
│     └─ code_presentation - the identification code of the presentation during which the student is registered on the module.
│     └─ id_student – a unique identification number for the student.
│     └─ gender – the student’s gender.
│     └─ region – identifies the geographic region, where the student lived while taking the module-presentation.
│     └─ highest_education – highest student education level on entry to the module presentation.
│     └─ imd_band – specifies the Index of Multiple Depravation band of the place where the student lived during the module-presentation.
│     └─ age_band – band of the student’s age.
│     └─ num_of_prev_attempts – the number times the student has attempted this module.
│     └─ studied_credits – the total number of credits for the modules the student is currently studying.
│     └─ disability – indicates whether the student has declared a disability.
│     └─ final_result – student’s final result in the module-presentation.
│
└─ studentRegistration.csv
│  └─ This file contains information about the time when the student registered for the module presentation. For students who unregistered the date of unregistration is also recorded. File contains five columns:
│     └─ code_module – an identification code for a module.
│     └─ code_presentation - the identification code of the presentation.
│     └─ id_student – a unique identification number for the student.
│     └─ date_unregistration – date of student unregistration from the module presentation, this is the number of days measured relative to the start of the module-presentation. Students, who completed the course have this field empty. Students who unregistered have Withdrawal as the value of the final_result column in the studentInfo.csv file.
│     └─ date_registration – the date of student’s registration on the module presentation, this is the number of days measured relative to the start of the module-presentation (e.g. the negative value -30 means that the student registered to module presentation 30 days before it started).
│
└─ studentAssessment.csv
│  └─ This file contains the results of students’ assessments. If the student does not submit the assessment, no result is recorded. The final exam submissions is missing, if the result of the assessments is not stored in the system. This file contains the following columns:
│     └─ id_assessment – the identification number of the assessment.
│     └─ id_student – a unique identification number for the student.
│     └─ date_submitted – the date of student submission, measured as the number of days since the start of the module presentation.
│     └─ is_banked – a status flag indicating that the assessment result has been transferred from a previous presentation.
│     └─ score – the student’s score in this assessment. The range is from 0 to 100. The score lower than 40 is interpreted as Fail. The marks are in the range from 0 to 100.
│
└─ studentVle.csv
│  └─ The studentVle.csv file contains information about each student’s interactions with the materials in the VLE. This file contains the following columns:
│     └─ code_module – an identification code for a module.
│     └─ code_presentation - the identification code of the module presentation.
│     └─ id_student – a unique identification number for the student.
│     └─ id_site - an identification number for the VLE material.
│     └─ date – the date of student’s interaction with the material measured as the number of days since the start of the module-presentation.
│     └─ sum_click – the number of times a student interacts with the material in that day.

For reference: https://www.kaggle.com/devassaxd/student-performance-prediction-complete-analysis
"""


def load_data():
    # load data into dataframes
    assessments = pd.read_csv(DATASET_DIR + "anonymiseddata/assessments.csv")
    courses = pd.read_csv(DATASET_DIR + "anonymiseddata/courses.csv")
    studAss = pd.read_csv(DATASET_DIR + "anonymiseddata/studentAssessment.csv")
    studInfo = pd.read_csv(DATASET_DIR + "anonymiseddata/studentInfo.csv")
    studVle = pd.read_csv(DATASET_DIR + "anonymiseddata/studentVle.csv")
    studRegis = pd.read_csv(DATASET_DIR + "anonymiseddata/studentRegistration.csv")
    vle = pd.read_csv(DATASET_DIR + "anonymiseddata/vle.csv")

    df = preprocess_data(assessments, courses, studAss, studInfo, studVle, studRegis, vle)

    return df


def preprocess_data(assessments, courses, studAss, studInfo, studVle, studRegis, vle):
    print("Preprocess data...")

    exams = assessments[assessments["assessment_type"] == "Exam"]
    others = assessments[assessments["assessment_type"] != "Exam"]
    amounts = others.groupby(["code_module", "code_presentation"]).count()["id_assessment"]
    amounts = amounts.reset_index()
    amounts.head()
    # Here we have the total amount of assessments by module

    # Function to determine whether a student passed a given assessment
    def pass_fail(grade):
        if grade >= 40:
            return True
        else:
            return False

    # Creating the stud_ass dataframe to join infos about the assessment weights and their respective grades
    stud_ass = pd.merge(studAss, others, how="inner", on=["id_assessment"])
    stud_ass["pass"] = stud_ass["score"].apply(pass_fail)
    stud_ass["weighted_grade"] = stud_ass["score"] * stud_ass["weight"] / 100

    # Final assessment average per student per module
    avg_grade = stud_ass.groupby(["id_student", "code_module", "code_presentation"]).sum()[
        "weighted_grade"].reset_index()
    avg_grade.head()

    # Pass rate per student per module
    pass_rate = pd.merge((stud_ass[stud_ass["pass"] == True].groupby(
            ["id_student", "code_module", "code_presentation"]).count()["pass"]).reset_index(), amounts, how="left",
                         on=["code_module", "code_presentation"])
    pass_rate["pass_rate"] = pass_rate["pass"] / pass_rate["id_assessment"]
    pass_rate.drop(["pass", "id_assessment"], axis=1, inplace=True)
    pass_rate.head()

    # Final exam scores
    stud_exams = pd.merge(studAss, exams, how="inner", on=["id_assessment"])
    stud_exams["exam_score"] = stud_exams["score"]
    stud_exams.drop(["id_assessment", "date_submitted", "is_banked", "score", "assessment_type", "date", "weight"],
                    axis=1, inplace=True)
    stud_exams.head()

    vle[~vle["week_from"].isna()]
    # Only 1121 from the 6364 entries have the reference week for the materials (the week in which they would be used in course.)
    # With this in mind, the construction of a metric to track study commitment becomes impractical

    studVle.head()

    # Here we can track the average time after the start of the course the student took to use the materials
    # and the average amount of clicks per material
    avg_per_site = studVle.groupby(["id_student", "id_site", "code_module", "code_presentation"]).mean().reset_index()
    avg_per_site.head()

    # General average per student per module
    avg_per_student = avg_per_site.groupby(["id_student", "code_module", "code_presentation"]).mean()[
        ["date", "sum_click"]].reset_index()
    avg_per_student.head()

    # Removing the cases where the student has withdrawn their registration to the module
    studInfo = studInfo[studInfo["final_result"] != "Withdrawn"]
    studInfo = studInfo[["code_module", "code_presentation", "id_student", "num_of_prev_attempts", "final_result"]]
    studInfo.head()

    df_1 = pd.merge(avg_grade, pass_rate, how="inner", on=["id_student", "code_module", "code_presentation"])
    assessment_info = pd.merge(df_1, stud_exams, how="inner", on=["id_student", "code_module", "code_presentation"])
    assessment_info.head()

    df_2 = pd.merge(studInfo, assessment_info, how="inner", on=["id_student", "code_module", "code_presentation"])
    final_df = pd.merge(df_2, avg_per_student, how="inner", on=["id_student", "code_module", "code_presentation"])
    final_df.drop(["id_student", "code_module", "code_presentation"], axis=1, inplace=True)
    final_df.head()
    # The final dataframe only has information relevant to the problem

    # final_df.describe()
    final_df.info()

    return final_df