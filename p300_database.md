# P300 Speller with patients with ALS

## Summary

This dataset represents a complete record of P300 evoked potentials recorded with BCI2000[1] using a
paradigm originally described by Farwell and Donchin [2]. In these sessions, 8 users with amyotrophic
lateral sclerosis (ALS) focused on one out of 36 different characters. The objective in this contest is to
predict the correct character in each of the provided character selection epochs.

## The paradigm

The user was presented with a 6 by6 matrix of characters (see Figure 1).The user’s task was to focus
attention on characters in a word that was prescribed by the investigator (i.e., one character at a time). All
rows and columns of this matrix were successively and randomly intensified at a rate of 4 Hz. Two out of 12
intensifications of rows or columns contained the desired character (i.e., one particular row and one
particular column). The responses evoked by these infrequent stimuli (i.e., the 2 out of 12 stimuli that did
contain the desired character) are different from those evoked by the stimuli that did not contain the
desired character and they are similar to the P300responses previously reported[2], [3].

```
Figure 1: User display for this paradigm
```
## Experimental Protocol

We included in the study a total of eight volunteers, all naïve to BCI training, ( 3 women; mean age= 58 ± 12 )
with definite, probable, or probable with laboratory support ALS diagnosis (mean ALSFRS-R scores: 32 ± 8
[4]). Scalp EEG signals were recorded (g.MOBILAB, g.tec, Austria) from eight channels according to 10– 10
standard (Fz, Cz, Pz, Oz, P3, P4, PO7 and PO8) using active electrodes (g.Ladybird, g.tec, Austria). All
channels were referenced to the right earlobe and grounded to the left mastoid. The EEG signal was


digitized at 256 Hz and band-pass filtered between 0.1 and 30 Hz. Participants were required to copy spell
seven predefined words of five characters each (runs), by controlling a P300 matrix speller. Rows and
columns on the interface were randomly intensified for 125ms, with an inter stimulus interval (ISI) of 125
ms, yielding a 250 ms lag between the appearance of two stimuli (stimulus onset asynchrony, SOA). For
each character selection (trial) all rows and columns were intensified 10 times (stimuli repetitions) thus
each single item on the interface was intensified 20 times.

Participants were seated facing a 15” computer screen placed at eye level approximately one meter in front
of them. The angular distance subtended by the speller was of 15 degrees. A single flash of a letter at the
beginning of each trial cued the target to focus. In the first three runs (15 trials in total) EEG data was
stored to perform a calibration of the BCI classifier. Thus no feedback was provided to the participant up to
this point. A stepwise linear discriminant analysis (SWLDA) was applied to the data from the three
calibration runs (i.e., runs 1–3) to determine the classifier weights (i.e., classifier coefficients). These
weights were then applied during the subsequent four testing runs (i.e., runs 4–7) when participants were
provided with feedback[5].

```
Table I: Demographic and clinical related data of
participants (N=8)
Age^ Sex^ ALSfrs-r^ Onset^
A01 56 M 13 Spinal
A02 59 M 37 Spinal
A03 43 M 33 Spinal
A04 75 F 38 Bulbar
A05 60 F 34 Bulbar
A06 40 M 31 Spinal
A07 61 M 28 Bulbar
A08 72 F 41 Bulbar
```
## Data set

X=[samples X Channels]

Y=[StimType X 1] ( 1 = NonTarget stimulus, 2 = Target Stimulus)

Y_stim= [StimClass X 1] intensified stimulus classes (Figure 2)

Trial=[Trials X 1] trial start in samples

Classes = textual description of conditions related to Y

Classes_stim = textual description of conditions related to Y_stim


```
Figure 2: this figure illustrates the assignment of the variable Y_stim to different row/column
intensifications
```
## References