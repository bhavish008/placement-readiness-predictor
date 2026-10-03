# DECISIONS

A log of the choices that changed the project, including what went wrong.

## 1. The dataset had no target column, so the label is rule-based

The provided file has 10 columns (`name`, `email`, `gender`, `age`, `course`,
`other_course`, `branch`, `year`, `cgpa`, `backlogs`). None of them says
whether a student is placement ready, so there was nothing to train on.

**Change:** a label `ready` was defined as CGPA of 7.0 or more **and** at most
1 backlog. This marks about 32% of students as ready.

**Consequence:** the models are relearning a rule that was written by hand.
The high scores show that the pipeline works, not that placement readiness can
be predicted from this data. With a real label the scores would be lower.

## 2. Failed approach: training on the raw 10,000 rows

The first run used all 10,000 rows. 9,000 of them are exact copies, leaving
only 1,000 unique students. After a normal train/test split, **100% of the
test rows also appeared in the training set**, so the test set was not unseen
data at all.

| Logistic Regression | F1 (ready class) |
|---|---|
| With duplicates (leaky) | 0.994 |
| Duplicates removed | 0.961 |

**Change:** duplicates are dropped before the split. The lower score is the
honest one. (Both figures are from the first feature set, which still
included `year`.)

## 3. Blank `backlogs` treated as "0", not as missing

`backlogs` is blank in about 27% of rows and the other values are `1`, `2`
and `3+`. There is no `0` anywhere, so blank is read as "no backlogs" and
kept as its own category instead of being imputed or dropped.

## 4. Columns left out

- `name`, `email`: identifiers.
- `other_course`: filled only when `course` is "Other", with 113 random values.
- `gender`: left out on purpose, since it should not decide readiness.

## 5. Why F1, precision and recall instead of accuracy

Only about 32% of students are ready. A model that always answers "not ready"
gets 68% accuracy while finding zero ready students (F1 = 0), which is what
the baseline row in the results shows.

## 6. Change: `year` removed from the features and the app

The first version asked for the student's year in the app. It was removed
because readiness, as defined here, is about academic standing (CGPA and
backlogs), and the year a student is in does not change that. The importance
check agreed: shuffling `year` changed F1 by 0.000.

Effect on the test set: Logistic Regression F1 went from 0.961 to 0.976, and
the Decision Tree stayed at 1.000. One fewer noise column gave the linear
model slightly less to be confused by.
