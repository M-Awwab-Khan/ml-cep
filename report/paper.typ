// Speech Emotion Recognition on CREMA-D: IEEE two-column paper version.
// Compile: typst compile --font-path report/fonts report/paper.typ report/paper.pdf
#import "@preview/charged-ieee:0.1.4": ieee

// Shared affiliation, written once.
#let dept = "Department of Software Engineering"
#let univ = "NED University of Engineering and Technology"
#let loc = "Karachi, Pakistan"
#let author(name, roll, email) = (
  name: name,
  department: [Roll no. #roll],
  email: email,
)

#show: ieee.with(
  title: [
    Speech Emotion Recognition from Audio Recordings: Low-Pass Noise Removal and Classical Spectral Features \
    #v(0.2em)
    #text(10.5pt)[Machine Learning (SE-317), Complex Engineering Problem \ Instructor: Miss Naveera Sami \
    #v(0.2em)
    #text(10.5pt, style: "italic")[#dept, #univ, #loc]]
  ],
  abstract: [We build and evaluate a pipeline that recognises six emotions (anger, disgust, fear, happiness, neutral, sadness) from 7,441 acted speech clips of the CREMA-D corpus. Exploratory analysis revealed two data issues that shaped the design: one clip is digital silence, and every clip carries an added 7.5 kHz tone holding about 95% of its energy, which a low-pass filter removes. Clips are split by *actor* (63/14/14) so that the test score measures generalisation to unseen speakers. From each filtered clip we extract 242 spectral features (MFCC, mel-spectrogram, chromagram, spectral centroid/bandwidth/contrast, long-term Fourier spectrum) plus eight bicoherence statistics, and compare five scikit-learn classifiers. The best models, logistic regression and an RBF SVM, reach 0.53–0.54 accuracy and 0.53 macro-F1 on unseen speakers (chance: 0.17), a plausible level for acted speech and classical features. Ablations show that 40 MFCCs carry almost all of the usable signal and that bicoherence adds nothing on top. We also report negative results (the filter cutoff barely changes accuracy; the added tone carries essentially no label information beyond clip loudness) and one unexplained observation, and discuss their limits.],
  authors: (
    author("Muhammad Awwab Khan", "SE-23051", "khan4600966@cloud.neduet.edu.pk"),
    author("Muhammad Abyaz Khalid", "SE-23053", "khalid4600054@cloud.neduet.edu.pk"),
    author("Nabeed Jamshed", "SE-23054", "ali4601063@cloud.neduet.edu.pk"),
    author("Muhammad Talha", "SE-23057", "talha4600894@cloud.neduet.edu.pk"),
  ),
  index-terms: ("Speech emotion recognition", "CREMA-D", "MFCC", "low-pass filtering", "bicoherence", "speaker-independent evaluation"),
  bibliography: bibliography("refs.bib"),
  paper-size: "a4",
  figure-supplement: [Fig.],
)

#let accent = rgb("#1f5fae")
#let muted = luma(105)
#show link: set text(fill: rgb("#1f5fae"))
#show ref: set text(fill: rgb("#1f5fae"))
#show cite: set text(fill: rgb("#1f5fae"))
#set par(justify: true)

// ---------- table style (booktabs-like, compact) ----------
#set table(stroke: none, inset: (x: 4pt, y: 2.6pt))
#let rtop = table.hline(stroke: 0.8pt)
#let rmid = table.hline(stroke: 0.5pt)
#let rbot = table.hline(stroke: 0.8pt)
#show table: set par(justify: false)

// ---------- helpers ----------
#let fmt(x, d: 3) = {
  let s = str(calc.round(float(x), digits: d))
  if not s.contains(".") { s += "." }
  let n = s.len() - s.position(".") - 1
  for _ in range(d - n) { s += "0" }
  s
}
#let mname = (
  LogReg: "Logistic regression", SVM: "SVM (RBF kernel)", NeuralNet: "Neural network (MLP)",
  KNN: "k-nearest neighbours", NaiveBayes: "Gaussian naive Bayes",
)
#let fin = csv("data/final_models.csv", row-type: dictionary)
#let cut = csv("data/cutoff_comparison.csv", row-type: dictionary)
#let abl = csv("data/ablation_single_group.csv", row-type: dictionary)
#let pick(rows, f, m) = rows.filter(r => r.features == f and r.model == m).first()


= Introduction

Automatic recognition of emotion from speech supports call-centre quality monitoring, media analytics and human–computer interaction. The task is hard because emotion is expressed through several interacting acoustic cues (energy, pitch, voice quality, timing) that also vary with speaker, sentence and recording conditions.

*Problem statement.* Given a short audio clip of an actor reading a sentence, predict which of six emotions was acted. The brief requires us to (i) select audio files and form training, validation and test sets, (ii) remove noise with a low-pass filter, (iii) extract features chosen from the literature, and (iv) evaluate K-nearest neighbours, logistic regression, naive Bayes, support vector machines and a neural network.

*Why this is more than a recipe.* The brief fixes the method families but leaves the decisions that determine whether the result is meaningful: how to split the data, what cutoff to use, which features to keep, and how to interpret scores. @tab:cep summarises how these decisions map onto the characteristics of a complex engineering problem.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    rtop,
    [Characteristic], [How it appears in this project],
    rmid,
    [Depth of knowledge], [Sampling and aliasing, IIR filter design, short-time spectral analysis, mel/cepstral processing, higher-order spectra, and statistical learning with proper validation.],
    [Conflicting requirements], [A low cutoff removes more noise but discards high-frequency speech cues; richer features help only if the ≈5,000 training clips can support them; flexible models overfit; bicoherence is costly to compute.],
    [Depth of analysis], [No prescribed split, cutoff or feature set: each is chosen and justified by experiment (cutoff sweep, per-group ablation, leakage analysis).],
    [Interdependence], [The cutoff alters every downstream feature; the split determines whether any reported score is meaningful; feature scaling interacts with every classifier.],
    rbot,
  ),
  caption: [Mapping of project decisions onto complex-engineering-problem characteristics.],
) <tab:cep>

*Contributions.* (1) A speaker-independent evaluation protocol with a justified split; (2) detection and characterisation of the dataset's added tone and of a silent clip; (3) a controlled study of filter cutoff and of each feature group, including bicoherence; (4) a comparison of five classifiers with validation-based tuning; (5) an error analysis in terms of emotional arousal, and a check that the added tone does not leak labels beyond clip loudness.

= Dataset and data quality

== CREMA-D

CREMA-D @cao2014 contains 7,442 clips from 91 actors (48 male, 43 female, aged 20–74, several ethnicities). Each actor read 12 sentences, each in one of six emotions (anger, disgust, fear, happiness, neutral, sadness) and, for one sentence, at four intensity levels. The copy used here stores mono WAV at 16 kHz; clips last 1.27–5.00 s (mean 2.54 s; 5.26 hours in total). Labels are encoded in the file name, e.g. `1001_DFA_ANG_XX.wav` = actor 1001, sentence DFA, anger, intensity unspecified. In the original study, human raters judging the *audio alone* identified the intended emotion in only about 41% of clips @cao2014, which gives a useful reference point for any classifier.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left,) + (right,) * 6,
    rtop,
    [Emotion], [Anger], [Disgust], [Fear], [Happy], [Neutral], [Sad],
    rmid,
    [Clips], [1,271], [1,271], [1,271], [1,271], [1,087], [1,271],
    rbot,
  ),
  caption: [Clips per emotion (all 7,442 clips). Intensity is unspecified for 6,077 clips (82%), so only the six emotions are used as labels.],
) <tab:classes>

== Data-quality findings

*A silent clip.* Scanning the peak amplitude of every file revealed that `1076_MTI_SAD_XX.wav` is exactly zero in all 41,108 samples. No model can learn from it and several statistics (e.g. skewness) are undefined, so it is excluded everywhere, leaving 7,441 clips.

*An added 7.5 kHz tone.* The data directory is named `NoiseAudioWAV`, and spectrograms of the original clips show a constant horizontal line at 7.5 kHz, even during pauses (@fig:noise a). We verified this with the FFT on 300 randomly chosen clips: the strongest spectral component lies within 7.4–7.6 kHz in 100% of them, and this band holds a median of 95% of the clip's total energy. The tone is therefore the dominant component of the raw signal and must be understood before any spectral feature (centroid, bandwidth, long-term spectrum) can be trusted. A low-pass filter whose cutoff is below 7.5 kHz removes it (@fig:noise b).

#figure(
  image("figures/fig_noise.png", width: 100%),
  caption: [One angry clip. (a) The original spectrogram shows the added 7.5 kHz tone as a bright line. (b) After a 5.5 kHz low-pass filter only speech remains. (c) Spectrum of the clip (0 dB = the tone) with the gain of the three filters studied; every cutoff suppresses the tone far below the speech level. Both spectrograms use the same colour scale.],
  placement: top, scope: "parent",
) <fig:noise>

== Clip selection

We use all 7,441 usable clips. The corpus is small enough that no subsampling is needed, the classes are nearly balanced (neutral is about 15% smaller), and discarding actors or sentences would only reduce the number of speakers available for the speaker-independent evaluation. The mild imbalance is handled by reporting macro-averaged F1.

= Methodology

#figure(
  {
    let step(t, b) = box(width: 100%, inset: (x: 4pt, y: 6pt), radius: 3pt, stroke: 0.7pt + accent, fill: accent.lighten(93%))[
      #align(center)[#set par(justify: false); #text(8.3pt, weight: "bold")[#t] \ #text(7.4pt, fill: muted)[#b]]
    ]
    let arr = align(center + horizon)[#text(accent)[→]]
    grid(
      columns: (1fr, 0.35cm, 1fr, 0.35cm, 1fr, 0.35cm, 1fr, 0.35cm, 1fr),
      align: horizon,
      step([Split by actor], [63 / 14 / 14 \ train / val / test]), arr,
      step([Low-pass filter], [Butterworth, order 5 \ forward–backward]), arr,
      step([Features], [242 spectral + 8 \ bicoherence]), arr,
      step([Standardise], [fit on train only]), arr,
      step([Classify], [5 models, tuned on val; scored once on test]),
    )
  },
  caption: [Processing pipeline. Every learned quantity (scaler, model parameters, hyperparameters) is fitted without access to the test actors.],
  placement: top, scope: "parent",
) <fig:pipeline>

== Speaker-independent split

Each actor appears in many clips, and a speaker's voice is a strong, emotion-irrelevant signature. If clips from one actor were spread over training and test sets, a model could exploit speaker identity and the test score would overstate real-world performance. We therefore assign whole actors to one split, drawn at random with a fixed seed: 63 actors for training (5,158 clips), 14 for validation (1,142) and 14 for testing (1,142). An assertion in the code confirms that no actor is shared between splits, and the emotion proportions are the same across splits. Hyperparameters and design choices are selected on the validation actors; the test actors are used only for the final reported scores.

== Low-pass filtering

A low-pass filter passes frequencies below a cutoff $f_c$ and attenuates those above. We use a Butterworth filter @butterworth1930 of order $n = 5$, designed with `scipy.signal.butter` and applied forwards and backwards (`filtfilt`) so that the output has zero phase delay. For a digital Butterworth filter the forward–backward amplitude response is

$ |H_"ff" (omega)| = 1 / (1 + (tan(omega slash 2) / tan(omega_c slash 2))^(2n)), $ <eq:butter>

where $omega = 2 pi f slash f_s$ is the frequency in radians per sample and $f_s = 16$ kHz. The tangent warping makes the cut-off much steeper near the Nyquist frequency (8 kHz) than the analogue formula suggests. Measured on real clips, the tone is reduced by about 61 dB for $f_c = 7$ kHz and by at least 147 dB for $f_c <= 5.5$ kHz, so every cutoff considered removes it. Because the cutoff trades noise removal against loss of high-frequency speech information (such as fricatives), we compare 3, 4, 5.5 and 7 kHz and an unfiltered baseline, reporting validation and test scores for each (@sec:cutoff explains how the setting used downstream was chosen).

== Feature extraction

*From a clip to a fixed-length vector.* Clips differ in length but classifiers need inputs of equal size. We (1) cut the signal into frames of 1,024 samples (64 ms) with a hop of 256 samples (16 ms), giving about 160 frames for a typical clip, and compute each feature per frame; (2) pool each per-frame coefficient over time by its mean and standard deviation. A feature with $C$ coefficients thus contributes $2C$ numbers, independent of clip duration. The mean captures the average sound; the standard deviation captures how much it varies, which differs between lively (angry, happy) and flat (sad, neutral) delivery. The cost is that temporal order is discarded.

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr),
    align: (left, right, left, left),
    rtop,
    [Feature], [Numbers], [What it measures], [Why it may separate emotions],
    rmid,
    [Fourier spectrum], [32], [Average magnitude per 250 Hz band over the clip (dB).], [Aroused speech carries more high-frequency energy.],
    [Mel-spectrogram], [128], [Energy in 64 mel-spaced bands per frame (dB, absolute level kept), mean and std.], [Preserves loudness and spectral shape on a perceptual scale.],
    [MFCC], [40], [First 20 cepstral coefficients: the smooth *shape* of the mel spectrum @davis1980.], [Reflects vocal-tract configuration and vocal effort.],
    [Chromagram], [24], [Energy in the 12 pitch classes.], [A crude proxy for pitch height and movement.],
    [Spectral centroid], [2], [Centre of mass of the spectrum ("brightness").], [Tense, aroused speech is brighter.],
    [Spectral bandwidth], [2], [Energy spread around the centroid.], [Strained voice spreads energy more widely.],
    [Spectral contrast], [14], [Peak-to-valley level difference in 6 octave bands plus residual.], [Separates clear voiced from breathy or noisy sound.],
    [Bicoherence], [8], [Mean, variance, skewness, kurtosis of the magnitude and of the phase of the bicoherence map.], [Nonlinear coupling of harmonics; may differ with vocal strain.],
    rmid,
    [*Total*], [*250*], [], [],
    rbot,
  ),
  caption: [Feature groups. The first seven give the 242-dimensional base vector; bicoherence is evaluated as an add-on.],
  placement: top, scope: "parent",
) <tab:features>

The mel scale warps frequency as $m = 2595 log_10(1 + f slash 700)$; MFCCs are the discrete cosine transform of the log mel energies, of which we keep the first 20.

*Bicoherence.* The power spectrum measures energy per frequency; the bispectrum @kim1979 measures whether components at $f_1$, $f_2$ and $f_1 + f_2$ keep a consistent phase relation, which indicates nonlinear interaction. For STFT frames $X_t (f)$ (256-point windows, hop 128), we estimate

$ B(f_1, f_2) = (E_t [X_t (f_1) X_t (f_2) overline(X_t (f_1 + f_2))]) / sqrt(E_t [|X_t (f_1) X_t (f_2)|^2] thin E_t [|X_t (f_1 + f_2)|^2]), $ <eq:bico>

and call $|B|$ (between 0 and 1) the bicoherence and $arg B$ the biphase. Over the grid $f_1, f_2 in {1, ..., 64}$ bins (62.5 Hz each) we summarise $|B|$ and $arg B$ by their mean, variance, skewness and kurtosis, giving eight numbers.

== Classifiers and tuning

All models are scikit-learn @pedregosa2011 estimators placed behind a `StandardScaler` fitted on training data only, because distance- and gradient-based models are sensitive to feature scale (centroid values are in the thousands, chroma values below 1). For each model we search the small grid in @tab:grid, train on the training actors, and keep the setting with the highest macro-F1 on the validation actors. The selected model is then scored once on the test actors. Features were extracted with librosa @mcfee2015. All random choices use seed 42.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    rtop,
    [Model], [Grid searched], [Selected (5.5 kHz, base features)],
    rmid,
    ..for (key, grid) in (
      ("KNN", [neighbours $k in {3, 7, 15, 31}$; uniform or distance weights]),
      ("LogReg", [inverse regularisation $C in {0.01, 0.1, 1, 10}$]),
      ("NaiveBayes", [variance smoothing $in {10^(-9), 10^(-6), 10^(-3), 10^(-1)}$]),
      ("SVM", [RBF kernel; $C in {1, 10, 100}$; $gamma in {$"scale"$, 10^(-3), 3 times 10^(-4)}$]),
      ("NeuralNet", [hidden layers $(128)$ or $(256, 128)$; L2 penalty $in {10^(-4), 10^(-2)}$; early stopping]),
    ) {
      (mname.at(key), grid, raw(pick(fin, "lp5500", key).best_params.replace("'", "").replace("{", "").replace("}", "")))
    },
    rbot,
  ),
  caption: [Hyperparameter grids and the values selected on the validation actors.],
  placement: top, scope: "parent",
) <tab:grid>

== Evaluation

We report accuracy and macro-averaged F1, the unweighted mean over the six classes of #box($F_1 = 2 P R slash (P + R)$) with precision $P$ and recall $R$. Macro-F1 prevents the smaller neutral class from being under-weighted. Chance level is $1 slash 6 approx 0.167$. With 1,142 test clips the binomial standard error of an accuracy near 0.53 is about 1.5 percentage points; because clips from the same actor are correlated, the true uncertainty is larger, so differences of one to two points are not meaningful.

= Results

== Effect of the filter cutoff <sec:cutoff>

#figure(
  {
    let tags = (("raw", "Unfiltered"), ("lp3000", "3 kHz"), ("lp4000", "4 kHz"), ("lp5500", "5.5 kHz"), ("lp7000", "7 kHz"))
    table(
      columns: (1fr, auto, auto, auto, auto, auto),
      align: (left,) + (right,) * 5,
      rtop,
      [Cutoff], [LR val], [LR test], [SVM val], [SVM test], [SVM acc.],
      rmid,
      ..for (t, l) in tags {
        let a = pick(cut, t, "LogReg")
        let b = pick(cut, t, "SVM")
        (l, fmt(a.val_f1), fmt(a.test_macro_f1), fmt(b.val_f1), fmt(b.test_macro_f1), fmt(b.test_acc))
      },
      rbot,
    )
  },
  caption: [Filter cutoff comparison: macro-F1 on validation and test actors for logistic regression (LR) and the SVM, plus SVM test accuracy (all 242 features).],
) <tab:cutoff>

All five settings give a test macro-F1 between 0.507 and 0.534, a spread of under three points that is comparable to the 1.5-point standard error; the data therefore do not support a claim that any cutoff is better. In particular, leaving the tone in does not hurt (unfiltered: 0.524 for logistic regression, 0.513 for the SVM). Every cutoff removes the tone, so this is not a failure of the filter. Two observations help explain it.

First, most of the features are insensitive to the tone: it adds a constant to a few frequency bands, which standardisation and the logarithmic mel scale largely absorb. The long-term Fourier spectrum and spectral contrast score identically with and without filtering (0.463 vs 0.460 and 0.410 vs 0.410 for an SVM on that group alone).

Second, the features that *average over the whole spectrum* are changed enormously: the mean spectral centroid is 5,562 Hz on unfiltered clips (pulled towards the tone) and 1,144 Hz on filtered clips (speech only). Unexpectedly, the unfiltered centroid and bandwidth then classify *better* than the clean ones (SVM macro-F1 0.335 vs 0.288 for the centroid, 0.338 vs 0.277 for the bandwidth). We did not establish why. A hypothesis, untested here, is that because the tone has a fixed level within a clip, the unfiltered centroid acts as a frame-by-frame measure of speech energy relative to that reference, a loudness-dynamics cue that the clean centroid lacks (see also @sec:leak). It is a reason to treat the tone as an artefact to be removed rather than a feature to be exploited, since real recordings would not contain it.

*Choice of cutoff.* We carried 5.5 kHz into the downstream experiments. In our first sweep (unfiltered, 3, 4 and 5.5 kHz) it ranked first for both models on the validation actors, and also on the test actors. When 7 kHz was added later, after we corrected an analytic estimate of the filter's attenuation (see Eq. @eq:butter), it scored nominally highest on validation (mean validation F1 0.511 against 0.507), but a difference of 0.004 is far inside the noise. We kept 5.5 kHz for consistency, which also preserves more of the speech band than 3–4 kHz, and report 7 kHz in @tab:cutoff. A strictly validation-only rule would select 7 kHz and give slightly lower test scores (0.520 and 0.507 macro-F1) without changing any conclusion.

#figure(
  image("figures/fig_cutoff.png", width: 100%),
  caption: [Test macro-F1 by filter cutoff for the two best models. Bars start at zero; the dashed line is chance.],
) <fig:cutoff>

== Which features matter? <sec:ablation>

To see which feature families carry the emotion signal, we trained an SVM on each group alone (@fig:ablation). The 40 MFCCs reach a macro-F1 of 0.524, indistinguishable from the 0.525 of all 242 features together, with about one-sixth of the dimensions. The 128-dimensional mel-spectrogram reaches 0.507, 1.7 points lower, which is within noise: the MFCCs, being a compact decorrelated summary of the same spectrum, do at least as well with a third of the dimensions, which matters when only ≈5,000 clips are available. The long-term Fourier spectrum (0.460) and spectral contrast (0.410, from only 14 numbers) are useful but weaker. The chromagram reaches only 0.338, as expected of a representation designed for music, where twelve pitch classes cannot describe speech intonation. Even single spectral-shape numbers carry information: the centroid (0.288) and bandwidth (0.277), two numbers each, are well above chance. Bicoherence alone reaches 0.322, about twice chance, so the eight statistics do carry signal; but, as shown in the next section, they add nothing to the other features. These are single-group results, not marginal contributions, and come from one split.

#figure(
  image("figures/fig_ablation.png", width: 100%),
  caption: [Test macro-F1 of an SVM trained on one feature group alone (5.5 kHz). Numbers in brackets are the group sizes.],
) <fig:ablation>

== Comparison of classifiers

#figure(
  {
    let order = ("LogReg", "SVM", "NeuralNet", "KNN", "NaiveBayes")
    let best = calc.max(..order.map(m => float(pick(fin, "lp5500", m).test_macro_f1)))
    table(
      columns: (1fr, auto, auto, auto, auto),
      align: (left,) + (right,) * 4,
      table.header([], table.cell(colspan: 2, align: center)[Base features (242)], table.cell(colspan: 2, align: center)[Base + bicoherence (250)]),
      table.hline(stroke: 0.5pt, start: 1, end: 3),
      table.hline(stroke: 0.5pt, start: 3, end: 5),
      [Model], [Accuracy], [Macro-F1], [Accuracy], [Macro-F1],
      rmid,
      ..for m in order {
        let a = pick(fin, "lp5500", m)
        let b = pick(fin, "lp5500+bico", m)
        let hi = float(a.test_macro_f1) == best
        (
          mname.at(m),
          if hi { strong(fmt(a.test_acc)) } else { fmt(a.test_acc) },
          if hi { strong(fmt(a.test_macro_f1)) } else { fmt(a.test_macro_f1) },
          fmt(b.test_acc), fmt(b.test_macro_f1),
        )
      },
      rbot,
    )
  },
  caption: [Test-set results on the 14 unseen actors (5.5 kHz low-pass). Chance is 0.167. Best base-feature macro-F1 in bold.],
) <tab:final>

Logistic regression and the RBF SVM are statistically indistinguishable and lead the table, followed closely by the neural network; k-nearest neighbours is clearly weaker and Gaussian naive Bayes weakest (@fig:models). Naive Bayes assumes independent features, which is badly violated by neighbouring, strongly correlated mel and cepstral coefficients. That a linear model matches a kernel SVM and a two-layer network indicates that the limiting factor is the information in the pooled features and the ≈5,000 training clips, not model capacity. Adding the eight bicoherence statistics does not improve any model (@tab:final): changes are within about two points and mostly negative.

#figure(
  image("figures/fig_models.png", width: 100%),
  caption: [Test macro-F1 of the five classifiers with and without bicoherence features.],
) <fig:models>

== Error analysis

#figure(
  image("figures/fig_confusion.png", width: 82%),
  caption: [Row-normalised confusion matrices on the test actors (each row sums to 1, so the diagonal is per-class recall).],
  placement: top, scope: "parent",
) <fig:confusion>

Anger is recognised best (79–81% recall for both models; @fig:confusion). Per-class results for logistic regression are given in @tab:perclass. The dominant confusions pair emotions that share a similar *arousal* (energy) level: happiness is mistaken for anger (17–20%) and fear (15–20%), and sadness for neutral (17–19%), fear and disgust. This is the expected failure mode of spectral features, which capture energy and spectral tilt well but carry little information about valence (pleasant versus unpleasant). Disgust has the lowest recall (42–44%); it is a subtle, low-arousal emotion that actors portray in diverse ways.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left,) + (right,) * 4,
    rtop,
    [Emotion], [Precision], [Recall], [F1], [Support],
    rmid,
    [Anger], [0.650], [0.790], [0.713], [195],
    [Disgust], [0.462], [0.436], [0.449], [195],
    [Fear], [0.500], [0.533], [0.516], [195],
    [Happy], [0.492], [0.472], [0.482], [195],
    [Neutral], [0.475], [0.515], [0.494], [167],
    [Sad], [0.648], [0.482], [0.553], [195],
    rmid,
    [Macro average], [0.538], [0.538], [0.534], [1,142],
    rbot,
  ),
  caption: [Per-class test results, logistic regression, 5.5 kHz, base features.],
) <tab:perclass>

== Does the added tone leak the label? <sec:leak>

Because the tone is an artefact added to the recordings, we checked whether it could act as a shortcut. Its amplitude correlates almost perfectly with the clip's peak level ($r = 0.99$), consistent with the tone being added at a level proportional to the clip's loudness. A logistic regression using *only* the tone amplitude reaches 0.369 test accuracy, about twice chance, but a regression using only the loudness (RMS) of the clean, filtered speech reaches the same 0.370, and the two are strongly correlated ($r = 0.86$). Using both together gives 0.394 (validation: 0.376, against 0.354 and 0.363 alone), a gain of one to two points that is within the noise, and the speech-to-tone ratio alone reaches only 0.253. We conclude that the tone carries essentially the same information as the clip's loudness, which is a genuine emotional cue (anger is loud, sadness is quiet) that the clean features already contain, and that any additional signal is small. This is consistent with filtering leaving accuracy unchanged for most features in @sec:cutoff. It does not explain the better classification by the unfiltered centroid and bandwidth reported there, which remains an open observation. Loudness is also partly an actor- and session-specific property, which is a reason to interpret absolute-level features with care.

= Discussion

*Performance in context.* About 53% six-class accuracy on unseen speakers is more than three times chance and above the roughly 41% reported for human listeners judging audio only @cao2014. Part of this edge is expected: the classifier is trained on this corpus's acting style and on speakers' recording conditions, whereas human raters were judging cold. We did not attempt deep models because the brief specifies classical classifiers, and make no claim about how they would compare.

*Why simple models suffice.* All feature groups are summaries of the same short-time spectrum, so they are strongly redundant. MFCCs compress that spectrum into a compact, decorrelated representation, which is why 40 of them match 242 features. Added capacity (kernel, hidden layers) has little to exploit, and the neural network is more prone to over-fitting at this sample size.

*Design lessons.* (i) The actor-wise split is the most consequential decision; a random clip split would almost certainly give a higher but misleading score. (ii) Reporting negative results matters: the cutoff and bicoherence experiments show that two components of the brief's pipeline do not move accuracy for these features, and the leakage check explains why. (iii) Checking assumptions pays off: an early analytic estimate of filter attenuation, based on the analogue Butterworth formula, was wrong by a wide margin for this digital filter and was replaced by measurement (see Eq. @eq:butter).

= Limitations and future work

- *Single split.* Results come from one random actor split with 14 test actors, so the confidence interval is wide. Repeating the protocol over several random or grouped cross-validation splits would quantify the variance.
- *Selection of the cutoff.* The 5.5 kHz setting was carried forward after a first sweep in which it also had the highest test score, so a small selection effect cannot be excluded; a strictly validation-based rule would pick 7 kHz (differences are within noise, @sec:cutoff). The ablation and final models use the single 5.5 kHz configuration.
- *Acted speech.* Performance on spontaneous, noisy, real call-centre speech will likely be lower.
- *Information lost by pooling.* Mean and standard deviation discard temporal order and intonation contours; sequence models or delta features could recover them.
- *Small hyperparameter grids* and a single feature configuration per ablation; interactions between groups were not exhaustively searched.
- *Bicoherence summarised by eight statistics.* Richer use of the 2-D map (for example band-wise means) might be more informative than the statistics prescribed here.
- *Loudness.* Absolute-level features may partly encode recording conditions; per-speaker normalisation is a natural extension.

= Conclusion

We built a reproducible, speaker-independent pipeline for six-class speech emotion recognition on CREMA-D. The data were found to contain a silent clip and an added 7.5 kHz tone, which a Butterworth low-pass filter removes. Classical spectral features with logistic regression or an RBF SVM reach about 0.53 accuracy and macro-F1 on unseen speakers; MFCCs carry nearly all of the signal, the filter cutoff and bicoherence features have no measurable effect on accuracy, and errors concentrate between emotions of similar arousal. The analysis framework (split, ablations, leakage check) is as much a result as the final number.

= Reproducibility

All code is in `src/` (`common.py`: data table, split, filter; `features.py` and `bicoherence.py`: features; `train_eval.py`: tuning and evaluation; `make_figures.py`: figures). The notebook `sentiment_analysis.ipynb` reproduces the analysis end to end from cached features. Experiments use Python 3.12, scikit-learn, librosa and SciPy with seed 42; this report is compiled with Typst 0.15 and reads its tables directly from the experiment CSV files.

