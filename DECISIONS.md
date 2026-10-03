# DECISIONS

A log of the choices that changed the project, including what went wrong.

## 1. The dataset had no target column, so the label is rule-based

The provided file has 10 columns (`name`, `email`, `gender`, `age`, `course`,
`other_course`, `branch`, `year`, `cgpa`, `backlogs`). None of them says
whether a student is placement ready, so there was nothing to train on.

**Change:** a label `ready` was first defined as CGPA of 7.0 or more **and**
at most 1 backlog (about 32% of students). This was later replaced by a
score formula, see decision 7.

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
honest one. (Both figures are from the first version: the simple rule label,
with `year` still a feature.)

## 3. Blank `backlogs` treated as "0", not as missing

`backlogs` is blank in about 27% of rows and the other values are `1`, `2`
and `3+`. There is no `0` anywhere, so blank is read as "no backlogs" and
kept as its own category instead of being imputed or dropped.

## 4. Columns left out

- `name`, `email`: identifiers.
- `other_course`: filled only when `course` is "Other", with 113 random values.
- `gender`: left out on purpose, since it should not decide readiness.

## 5. Why F1, precision and recall instead of accuracy

Only about 23% of students are ready. A model that always answers "not ready"
gets 77% accuracy while finding zero ready students (F1 = 0), which is what
the baseline row in the results shows.

## 6. Change: `year` removed from the features and the app

The first version asked for the student's year in the app. It was removed
because readiness, as defined here, is about academic standing (CGPA and
backlogs), and the year a student is in does not change that. The importance
check agreed: shuffling `year` changed F1 by 0.000.

Effect on the test set (with the simple rule label of decision 1): Logistic
Regression F1 went from 0.961 to 0.976, and the Decision Tree stayed at 1.000. One fewer noise column gave the linear
model slightly less to be confused by.

## 7. Change: label now comes from a readiness score, not a flat rule

The flat rule treated a student with CGPA 7.0 and one backlog the same as a
student with CGPA 9.5 and none. A score lets the two features trade off.

- Not eligible (CGPA below 7.0, or more than 1 backlog): score 0.
- Eligible: score = 60 + (CGPA - 7) / 3 x 40 - 15 x backlogs.
- Ready = score of 65 or more.

This works out to CGPA 7.38 or more with no backlogs, or CGPA 8.5 or more
with one backlog. Ready students dropped from about 32% to about 23%.

**The weights are a judgement call.** 60, 40, 15 and 65 are not derived from
data, because the data has no placement outcomes to derive them from.

**Rejected alternative:** predicting the score itself. That is regression, and
the task asks for binary classification.

Effect on the test set: Logistic Regression F1 is 0.989 (it misses one ready
student out of 46), and the Decision Tree is still 1.000. The label is still
deterministic, so high scores are still expected.

## 8. Change: probability removed from the app's result

The app first showed the model's probability next to the formula score, for
example "Ready for placement (100% probability)" above "72 / 100". Two
different numbers for the same student were confusing: the 100% was the
Decision Tree's confidence, and the 72 was the formula's score.

**Change:** the app now shows only the verdict and the score. The verdict
line names the model ("Decision Tree says: Ready for placement") so it is
clear that the verdict comes from the trained model and the score comes from
the formula.

**Trade-off:** the probability was the only sign of how sure the model was.
Logistic Regression's borderline cases (for example 50.1%) are no longer
visible in the app. They are still printed by `train.py`.

## 9. Addition: warning when the model and the formula disagree

With the probability gone, the app could show "Ready" beside a score below
65 with no explanation. This happens with Logistic Regression for students
just under a cutoff, for example CGPA 8.4 with one backlog (score 64, model
says ready) or CGPA 7.3 with no backlogs (score 64, model says ready).

**Change:** the app shows a short warning whenever the model's verdict and
the formula's verdict differ. The Decision Tree did not disagree with the
formula in any case tested.
