# Ball possession frame by frame (BAR-ATH)

A solution to the Ball Possession task: for each of the 199,319 frames of the tactical video, a
label `H` (the home team has the ball), `A` (the away team), `D` (ball out of play) or `X`
(outside the halves).

Possession is determined from the tactical-camera tracking alone. The models are trained on
open labelled matches, and the settings are chosen on open matches as well; the labels of the
competition match are not used anywhere.

| | |
|---|---|
| submission file | `output/submission.csv` |
| how to get it | `run_all.sh` — download the open data, train the models and label the match, about 40 minutes ([section 4](#4-reproduction)) |

---

## 1. Approach

The ranking metric `spellF1` scores whole possessions, and frame-by-frame decisions flicker and
split them. So the solution has two parts: per-frame *evidence* ("whose ball", "is the ball in
play") and a *decoder* that changes the label only when the evidence outweighs the cost of a
change.

The tracker does not see the ball on 41% of the graded frames, so besides the model that reads
the ball there are models that read only the players. The competition match has no labels to
train on: the models are trained on seven open DFL matches, and the decoder settings are chosen
on three open matches from Metrica, the provider of the competition's ground truth.

1. **Tracker profile.** Measured from the coordinates of the competition match, without labels:
   how smooth the tracks are and how often the tracker loses players and the ball.
2. **Degrading the external matches.** The open matches are corrected tracking: every player
   and the ball are always there. Before training, the tracks are smoothed, and players and the
   ball are hidden as often as the competition's tracker loses them.
3. **Features:** 132 from the players (position, shape and movement of the teams) and 67 with
   the ball (distances to the nearest players, speed and position of the ball). The two teams
   are treated symmetrically.
4. **Three gradient-boosting models** (own implementation in numpy; an ensemble of five, 300
   trees of depth 4): "who has the ball" from the players, "is the ball in play" from the
   players, "who has the ball" from the players and the ball. Frames of short possessions weigh
   more.
5. **Three temporal convolutional networks** (64 channels, a view of ±15.6 s): "who has the
   ball" from the same features. The network places the moment the ball changes hands more
   precisely: with it the best `spellF1` on Metrica is 0.666, without it 0.633. Training is in
   PyTorch, inference in numpy.
6. **Decoder settings** — seven numbers, one of them the share of the network in the answer
   (0.25). All 62,720 combinations of the grid are tried on the Metrica matches and scored with
   the official metric. The choice is not the best combination but the middle of the plateau of
   60 nearly equal ones.
7. **Decoding the match** (Viterbi in two passes): first "in play / out of play", then, inside
   each stretch of play, which team. Where the laws of the game give certainty, rules decide
   ([section 3](#3-possession-rules)).

## 2. Data

### 2.1. Competition data

| asset | how it is used |
|---|---|
| tactical tracking (ATD): players and ball | the main input |
| Smart Tagging | the starts of the halves (the Kick Off tag) and the starts of set pieces: throw-in, corner, goal kick, free kick. The phases of play with a team label are not used: they are phases, not possessions |
| broadcast tracking | tried and dropped. After aligning the two time scales, without labels: the ball is visible in it on 48% of the graded frames, in the tactical tracking on 59%, in at least one of the two on 63%. So it would add the ball on fewer than 4% of the frames, and on a fifth of the frames where both trackers see the ball they disagree by more than 2.5 m |
| video (tactical and broadcast), homography | not used |
| possession ground truth, including the public five minutes | not used; the code does not read it |

### 2.2. External data

| dataset | what it is | licence | source |
|---|---|---|---|
| DFL / IDSSE | 7 matches of the Bundesliga and 2. Bundesliga, tracking at 25 frames/s with possession and ball-status flags. Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12, 195, doi:[10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y) | CC-BY 4.0 | `huggingface.co/datasets/pysport/idsse-data` |
| Metrica Sports Sample Data | 3 anonymised matches: tracking at 25 frames/s and events | open data; the authors ask that the source be credited | `github.com/metrica-sports/sample-data` |

DFL is used to train the models and the networks. Metrica is used only to choose the decoder
settings. Both datasets are downloaded by step 1; the checksums of all 23 files are in
`external_data.sha256`.

## 3. Possession rules

The labels follow the conventions of the providers whose data the models were trained on and
the settings were chosen on: possession belongs to the team that last controlled the ball, for
as long as the ball is in play.

**Ball in play or not (`D`)**

* The main evidence is the behaviour of the players: during stoppages the tracker hardly sees
  the ball. The state changes only when the evidence adds up to more than the penalty for a
  change, so a one-second lull does not become a stoppage.
* Players stop with a delay, so a stoppage found from the players starts 22 frames earlier —
  unless its start was fixed by the ball itself.
* The ball itself proves a stoppage when it is seen more than a metre beyond the boundary line;
  when it lies still for two seconds with nobody within 1.5 m (a placed ball); when it lies in
  the goal area and for a quarter of that time nobody is near it (a goal kick).
* A restart needs a dead ball. If there is no stoppage near a throw-in, corner, goal-kick or
  free-kick tag (from 5 seconds before the tag starts to 8 seconds after), the first 3.5
  seconds of the tag are treated as a stoppage. If it is a goal kick and a "goalkeeper has the
  ball" episode falls on it, the episode is taken to be the placing of the ball and becomes a
  stoppage too.
* The frames of a half before the Kick Off tag starts are `D`.
* Penalty: `D` while the players are lined up; from the kick the ball is in play.

**Whose ball (`H` / `A`)**

* A loose ball, a challenge or a rebound does not change possession. The label passes to the
  opponent when the evidence in their favour has built up and outweighed the penalty for a
  change, that is, when they have taken control of the ball.
* The last touch before a stoppage: a possession that ends in a stoppage has to be confirmed
  especially firmly (ten times the penalty for a change). Otherwise it is a touch after which
  the ball went out (a defender cleared it into touch), and the frames stay with the team that
  had the ball before it.
* Goalkeeper holding the ball: the ball is in play and belongs to the goalkeeper's team, even
  if the tracker has lost the ball and nothing is happening on the pitch. This differs from a
  goal kick in that the ball does not lie still in the goal area.
* After a penalty kick the ball belongs to the kicking team for the time of its flight to the
  goal; after that the evidence decides.
* Inside the halves `X` is never used: it is always `H`, `A` or `D`.

## 4. Reproduction

The folder has to be in the root of the competition repository, next to `data/`, `evaluation/`
and `games.py`. All commands are run from the repository root.

**Requirements**

| | |
|---|---|
| competition data | `data/tactical/BAR-ATH_tactical_atd.txt`, `…_atd.xml`, `…_smart_tagging.xml` |
| repository files | `games.py`, `evaluation/validate_submission.py`, `evaluation/possession_metrics.py` |
| Python | 3.9 (tested with 3.9.6) |
| libraries | `numpy` 2.0.2, `pandas` 2.3.3, `torch` 2.8.0 — from `requirements.txt`; `torch` is needed only by step 3 |
| internet | only for step 1 (2.8 GB of open data) |
| disk | about 6 GB during training |
| memory | about 2 GB per process |

No GPU is needed: the networks are small and train on the CPU.

**Environment**

```bash
python3 -m venv solution/.venv
```

```bash
solution/.venv/bin/pip install -r solution/requirements.txt
```

**Run** — download the open data, train the models, choose the settings, label the match:

```bash
bash solution/run_all.sh
```

`run_all.sh` runs the four steps in order; each of them can also be run on its own.

| step | script | what it does | what appears | time* |
|---|---|---|---|---|
| 1 | `step1_download.py` | downloads 7 DFL matches and 3 Metrica matches, verifies the checksums | `external/` | ~10 min |
| 2 | `step2_train_boosting.py --jobs 5` | measures the tracker profile, trains the three boosting models on DFL, applies them to the Metrica matches | `trained/target_profile.json`, `trained/models.npz` | ~12 min |
| 3 | `step3_train_network.py --jobs 5` | trains the three networks on DFL, chooses the decoder settings on Metrica | `trained/seq_team.npz`, `trained/decoder.json` | ~18 min |
| 4 | `step4_predict.py` | labels the competition match, checks the file with the organisers' validator | `output/submission.csv` | ~1.5 min |

\* On an Apple M5 Pro with seven processes (`JOBS=7 bash solution/run_all.sh`). The number of
processes affects only the speed.

The `trained/` folder with the trained models appears after steps 2 and 3, and the submission
file `output/submission.csv` after step 4. Once the models are trained, the match can be
labelled again with step 4 alone (2 minutes, only numpy is needed):

```bash
solution/.venv/bin/python solution/step4_predict.py
```

**How exact the reproduction is.** All randomness is seeded. On macOS arm64 a full run from
scratch gives byte-identical models and the same submission file. Step 4 with the same models
was also checked on x86-64 (under Rosetta emulation) with two builds of numpy, Accelerate and
OpenBLAS: the file is the same. It was not run on Linux. Retraining on another platform was not
checked either: floating-point sums may differ in the last bits, and then the models, and with
them a few frames of the submission, may differ.

## 5. Files

```
solution/
  README.md                  this file
  requirements.txt           numpy, pandas, torch (exact versions)
  run_all.sh                 all the steps in order
  step1_download.py          step 1: the open data
  step2_train_boosting.py    step 2: tracker profile and boosting
  step3_train_network.py     step 3: networks and decoder settings
  step4_predict.py           step 4: labelling the match
  external_data.sha256       checksums of the external data
  possession/                the code
    build.py                 the four steps as functions
    dataio.py                reading the match tracking and tags, writing the submission file
    adapt.py                 tracker profile and degrading the external matches
    idsse.py, download.py    DFL: reading and downloading
    metrica.py               Metrica: downloading, reading, building the labels
    features.py              frame features
    gbdt.py, models.py       gradient boosting in numpy and the three models
    nn.py, nn_train.py       temporal network: inference (numpy) and training (PyTorch)
    validate.py, tune.py     the Metrica matches as the models see them; choosing the decoder settings
    labelling.py, decode.py  evidence → labels: ball rules and Viterbi
    keeper.py, penalty.py    goalkeeper and penalty rules
    tags.py                  set-piece rule
    scenes.py, pipeline.py   the competition match from tracking to labels
  trained/                   appears after a run: tracker profile, models, networks, decoder settings
  output/submission.csv      appears after a run: the submission file
```
