% =========================================================================
% GCH 2026 full-paper draft — "Reintegrating losses in heritage embroidery"
% -------------------------------------------------------------------------
% NOTES FOR OVERLEAF:
%  * Written against the Eurographics egpubl class used by the official GCH template. Start from the official GCH 2026 Overleaf template and paste everything from \title{} down into it; the preamble below mirrors the template but the class options change year to year.
%  * Submission is DOUBLE-BLIND: leave the anonymous author block as-is.
%  * \TODO{...} marks gaps/place-fillers. \fignote{...} marks figures that still need assembling.
%  * All quantitative numbers in Section 6 are from the scale-matched scoring run of 2026-06-05 (scores_sm_*.csv); cross-check against the final CSVs before submission.
%  * Paragraphs are single long lines by design (no hard wraps).
% =========================================================================
\documentclass[manuscript]{acmart}

\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{xcolor}

\newcommand{\TODO}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\fignote}[1]{\textcolor{purple}{[FIGURE: #1]}}
\newcommand{\PRELIM}{\textcolor{orange}{\textbf{[PRELIMINARY]}}}
\newcommand{\degr}{\ensuremath{^\circ}}
\graphicspath{{figures/}}

\title[Reintegrating Losses in Heritage Embroidery]{Digitally Reintegrating Losses in Heritage Embroidery with Model-Independent Stitch Priors and Surface-Normal Verification}

% --- double blind: do not fill in ---
\author[Anonymous]{Anonymous submission}

\begin{document}

\maketitle

% =========================================================================
\begin{abstract}
Embroidered heritage textiles commonly survive with losses: regions where thread has worn, broken, or been removed, leaving the ground fabric or earlier working exposed. Physical reintegration of such losses is slow, costly, and in many cases ethically precluded; yet a credible image of the intact object is valuable for conservation planning, interpretation, and public engagement. We present a workflow for digital reintegration of losses in embroidered textiles that is deliberately independent of any single generative model. Lightweight low-rank adapters (LoRAs) are trained to encode individual stitch structures (satin stitch, french knot, and silk purl) from photographic crops of 17th-century English embroideries, and are deployed on the inpainting or instruction-edit variants of four contemporary image-generation model families. Because a faithful fill should reproduce the craft surface structure of a stitch type rather than the exact thread positions of the lost original, we verify fills in surface-normal space: a monocular normal estimator (Marigold) fine-tuned on 11,813 reflectance-transformation-imaging (RTI) tiles of the same corpus converts each fill to a normal map, from which seven training-free surface descriptors are computed and compared, by Mahalanobis distance, to the descriptor cloud of real stitches of the prescribed type. On synthetic losses cut into intact held-out tiles, fills produced without the stitch LoRA drift steadily away from the real-stitch cloud as the loss grows, while fills produced with the LoRA remain statistically indistinguishable from real embroidery at every scale tested, across all six model variants and all three stitch types evaluated. The workflow requires no architecture-specific components, runs end-to-end on a single consumer GPU, and is intended to designed to be robust to the rapid churn of the underlying generative models.
\end{abstract}

% =========================================================================

\section{Introduction}
\label{sec:intro}

Recent advances in generative modelling, particularly through diffusion-based frameworks, have significantly enhanced the synthesis of visually plausible imagery across diverse domains. State-of-the-art systems are now highly effective in domains that are well represented in their training data: portraiture, nature scenes, cinematic composition, where annotations are abundant and stylistic features broadly consistent, and they are optimised above all to convincingly produce outputs indistinguishable from photographs. In cultural heritage contexts, however, this emphasis on photorealism introduces a conceptual pitfall: photorealism is not synonymous with authenticity, and a generated image that appears visually plausible may still fail to reflect the material and historical truth of an artefact~\cite{Zhang2024}.

The distinction is acute for embroidered textiles, which are among the most fragile classes of heritage object. Thread loss, abrasion, insect damage, and deliberate re-working leave many surviving pieces with losses: bounded regions in which the original stitched surface is absent and the ground fabric, underdrawing, or a later repair shows through. In conservation practice the treatment of losses is governed by the well-established principles of minimal intervention, reversibility, and the distinguishability of any added material from original work~\cite{Brandi1963,MunozVinas2005}. Manual techniques such as darning, couching, and the application of support fabrics continue to underpin textile conservation~\cite{Grimm1995}, but for many embroidered objects these principles rule out physical reintegration altogether: re-stitching a loss risks damaging adjacent original thread, an intervention sympathetic enough to read correctly demands scarce craft skill in techniques that are themselves endangered, and a speculative reconstruction risks asserting a history the evidence cannot support.

Digital reintegration offers a way through this impasse. A digital fill costs the object nothing, is trivially reversible, and can be presented alongside the original state, to support conservation planning (source), interpretation of damaged and unfinished works (source), and forms of public engagement that the damaged original cannot sustain~\cite{Jenkinson2025}. Generative models bring genuine capability to this task: their capacity to infer occluded content and extrapolate structural continuities is precisely what loss reintegration demands. GAN-based virtual restoration of paintings has produced contextually coherent inpainting of cracks and losses, outperforming traditional patch-based methods~\cite{Sizyakin2021}, and domain-adapted diffusion models have maintained symmetry, pattern continuity, and stylistic consistency in the restoration of heritage pottery~\cite{Zhang2024}. A model completing a damaged embroidery can, in principle, continue motifs logically from symmetry and learned structure while maintaining realistic texture grounded in domain-relevant stitch types, even where no explicit design reference survives.

This kind of extrapolation mirrors the embodied reasoning employed by skilled craftspeople. In textile conservation, practitioners reconstruct losses by analysing weave structures, stitch techniques, and pattern repetition, drawing on extensive tacit knowledge; the expert restorer evaluates both the structural and aesthetic logic of an artefact and the feasibility of reconstructing its form using historically appropriate methods~\cite{Brandi1963,MunozVinas2005}. Any computational tool that aspires to assist this work should be judged by the same standard: not whether its output is aesthetically pleasing, but how well it adheres to the heritage construction logic of the craft.

General-purpose generative models do not meet this standard out of the box. They inherit their understanding of images from vast but generalised datasets in which specialised craft surfaces are at best under-represented, and they are not trained to respect the physical constraints or stylistic idiosyncrasies of heritage artefacts; in practice this yields reconstructions that are visually smooth but materially implausible. The pattern recurs across heritage media: virtual restoration of ancient murals requires domain-specific structural guidance before standard inpainting becomes adequate~\cite{Ge2023}, and conventional inpainting applied to damaged silk relics produces blurry or unnatural results precisely where structural information is missing from the damaged region~\cite{Sun2023}. For textiles specifically, a growing body of work argues that encoding weave structure, yarn properties, and pattern-repetition conventions is what makes computational reconstruction credible: surveyed techniques for virtually reconstructing fragmented archaeological textiles foreground exactly these properties~\cite{Gigilashvili2023}, perceptual-loss inpainting has been tailored to traditional textile motifs~\cite{Stoean2022}, convolutional classification has been used to guide the digital repair of ancient textiles~\cite{Sha2024}, and generative methods have been explored for encoding craft-specific structural relationships in lace~\cite{Ahteck2024}. Ding and Liang's review of digital restoration for heritage clothing additionally notes that translating computational proposals into physically achievable interventions remains largely unexplored~\cite{Ding2024}, while surveys of machine learning in cultural heritage repeatedly identify the gap between general-purpose models and domain-specific validity as the persistent obstacle~\cite{Fiorucci2020}.

What is missing from this landscape is a standard of evidence. A fill can be judged plausible in RGB while being wrong as craft: embroidery is a relief medium, and its identity lies in thread direction, stitch geometry, and surface structure that colour similarity only weakly constrains. Concerns regarding over-reliance, deskilling of professionals, and the prospect of AI-generated content being mistaken for authentic heritage material corroborate this ~\cite{Spennemann2024}, and AI-assisted restorations are accordingly held to require expert evaluation for cultural and historical accuracy~\cite{Zhang2024}. If digital reintegration is to earn a place in conservation practice, the question of whether a fill is faithful to the craft must be answerable with robust measurement.

This paper contributes a workflow in which a conservator delineates a loss, prescribes the stitch technique that the surviving evidence indicates, and receives a digital reintegration based on the surviving parts of the artefact (or samples of closely related work) together with an objective account of its fidelity. We propose that this fidelity should be measured by a score grounded in the measured surface statistics of real examples of that technique, decomposable into interpretable terms that a specialist can interrogate and overrule. Such verified proposals could support treatment planning before any needle touches thread, drive interpretation and engagement presentations of damaged or unfinished works, and because they are verified in surface-geometry space, eventually guide physical realisation, whether by hand or by computerised embroidery~\cite{Ding2024, Jenkinson2025}.

This workflow this paper produces is built around three ideas.

\textbf{Stitch structure encoded as a transferable LoRA.} We train small low-rank adapters (LoRAs)~\cite{Hu2022} that encode individual stitch types (satin stitch, French knot, and silk purl) from photographic crops of 17th-century English embroideries. Because these LoRAs are lightweight and act on the shared attention layers of their base models, the same dataset and training recipe applies across four contemporary generative families (SDXL, FLUX.1, Qwen-Image, Z-Image), and each trained LoRA carries over onto its family's inpainting or instruction-edit variant. The workflow is therefore a method generalisable across models: training a stitch LoRA takes in the order of hours, so when a base model is inevitably superseded the same crops, captions, and hyperparameters simply re-create the stitch vocabulary on the successor, and the verification protocol carries over unchanged to benefit form the characteristics of a newer model or one with a different architecture.

\textbf{Verification in surface-normal space.} Embroidery is a relief medium; its identity lies in surface structure that RGB similarity only weakly constrains. We fine-tune the Marigold monocular normal estimator~\cite{Ke2024,Ke2025} on RTI-derived normal maps of the same corpus, validate it against RTI ground truth (mean angular error on high-relief tiles roughly halves, e.g.\ from 27\degr{} to 13\degr{}), and then use it as a calibrated instrument: every generated fill is converted to a normal map and evaluated on the surface geometry it depicts.

\textbf{An objective, repeatable, robust fidelity metric.} A faithful fill is not a reconstruction. The lost stitches' exact positions are unknowable, such that a well-conditioned model will generate fills that may differ from the original. For this niche purpose, our generative step is evaluated on whether the fill is a statistically realistic example of the prescribed stitch type. Seven training-free surface descriptors (slope statistics, roughness, bump density and spacing, directional coherence, dominant spatial pitch) are computed from the fill's estimated normals and compared, by Mahalanobis distance, to the descriptor distribution of real stitches of that type. The descriptor space cleanly separates the three stitch types on real data (95\% nearest-centroid accuracy {TODO - check this is still the case}), so distance to a type's cloud is a meaningful measure of craft fidelity, independent of where individual stitches happen to fall.

Because the metric requires real normals for the masked region only at validation time, we obtain fully controlled ground truth by cutting synthetic losses into intact held-out tiles. This yields the paper's central quantitative result (Section~\ref{sec:results}): without the stitch LoRA, fill fidelity degrades steadily as the loss grows; with the LoRA, fills remain within the variability of real embroidery at every loss size tested, across all six model variants evaluated, and the benefit of the LoRA grows with the size of the loss, precisely the regime where reintegration is hardest and most valuable.

We position this work within conservation ethics rather than against them: nothing here touches the object, every output is labelled and reversible, and the verification protocol gives conservators an evidential basis for deciding whether a digital reintegration is faithful enough to show. Through the integration of physical logic, historical context, and craft feasibility, the aim is to demonstrate that generative AI can augment expert-led restoration practice and enhance opportunity for engagement. Code, demonstration material, and a step-by-step how-to accompany the paper on a project page \TODO{anonymised link for review; live page on acceptance}.

% =========================================================================
\section{Related Work}
\label{sec:related}

\textbf{Digital restoration and inpainting in cultural heritage.} Virtual reintegration of lacunae is established in paintings and frescoes: deep-learning inpainting of cracks and losses outperforms patch-based methods on paintings~\cite{Sizyakin2021}, dedicated networks with structural guidance restore ancient murals~\cite{Ge2023}, and LoRA-adapted diffusion models reconstruct painted pottery~\cite{Zhang2024}. Textiles are comparatively under-served: existing work spans virtual reconstruction of fragmented archaeological textiles~\cite{Gigilashvili2023}, structure-guided restoration of silk relics~\cite{Sun2023}, perceptual-loss inpainting of traditional motifs~\cite{Stoean2022}, CNN-guided repair of ancient textiles~\cite{Sha2024}, and generative encoding of craft structure in lace~\cite{Ahteck2024}; Ding and Liang review the field for heritage clothing~\cite{Ding2024}. Across all of these, evaluation is conducted in image space which cannot assess whether a reconstruction respects the surface structure of the craft.

\textbf{Generative texture priors and LoRA fine-tuning.} Low-rank adaptation~\cite{Hu2022} has become the standard mechanism for injecting narrow visual concepts into large diffusion models, alongside subject-driven fine-tuning~\cite{Ruiz2023} and textual inversion~\cite{Gal2023}. We use LoRAs not for subjects or styles but for craft structures, and exploit their transferability between a base model and its inpainting fine-tune.

\textbf{RTI and photometric capture of textiles.} Reflectance transformation imaging is established for documenting surface relief of heritage objects~\cite{Malzbender2001} \TODO{cite textile-specific RTI applications}, and photometric-stereo-derived normals have been used for stitch-level analysis \TODO{Xavi's paper?}.

\textbf{Monocular surface-normal estimation.} Diffusion-based geometry estimators, notably Marigold~\cite{Ke2024,Ke2025}, repurpose image priors for dense prediction and fine-tune effectively on modest domain-specific data. We contribute an embroidery-specialised fine-tune and contribute the idea of using the estimator as a measurement instrument inside an evaluation protocol.

\textbf{Evaluating generative fills.} Standard inpainting metrics (PSNR/SSIM against the original, FID against a corpus) either assume reconstruction or require learned feature spaces \TODO{cite}. Our descriptor-space approach is closest in spirit to classical texture analysis \TODO{cite: structure tensor, spectral texture features} and avoids stacking a second learned model on top of the estimator being verified.

% =========================================================================
\section{Materials and Data}
\label{sec:data}

\subsection{Corpus and capture}

The corpus comprises N=7 17th-century English embroidered textiles (partnership with Holburne declaration?). Each piece was captured by colour reflectance transformation imaging in 42 megapixels ($7956 \times 5312$) tile captures, with surface normals recovered per capture by combining RTI with photogrammetry (reference? short summary?). The result is a set of registered (colour, normal-map) image pairs covering both intact and damaged regions of each piece.

For training the normal estimator the captures were tiled on a regular $768$-px grid (interior tiles flush, edge tiles shifted inward), giving $\approx$12,000 colour--normal tile pairs. The estimator is trained and evaluated at $768$ px to match Marigold's base training corpus.

\subsection{Stitch-type datasets}

Three stitch types were selected to span a broad subset of the structural range of the corpus: satin stitch (long parallel floats; smooth, strongly directional), French knots (discrete raised bumps with distinct thread features linked to technique), and silk purl (coiled metal-wrapped thread; fine-pitch helices with significant relief). For each type we cropped regions of homogeneous stitching from the colour captures at native resolution ({n}=40 for each stitch at $1024^2$). Each crop was captioned with a short structural description containing a rare trigger token per type (\texttt{embstn}, \texttt{embfnchknt}, \texttt{embslkprl}) so that the learned visual representation of the stitch type can easily be invoked or withheld at inference inheriting minimal bias from pre-existing significance attached to words already within the transformer's corpus (e.g. "french", "satin", "silk").

\subsection{Held-out evaluation tiles and synthetic losses}

96 intact tiles spanning all three stitch types (32 of each type) were held out of all training (both LoRA and normal-estimator training) for a quantitative test set, each cropped at native $1024^2$ directly from its source capture. Synthetic losses were generated by applying centred square masks at three sizes: $128$, $256$, and $512$ px, such that the true surface of every masked region is known, and intact visual context survives around the synthetic loss for continuity reference. A separate set of genuinely damaged and unfinished pieces is reserved for qualitative demonstration (Section~\ref{sec:results-real}).

% =========================================================================
\section{Method}
\label{sec:method}

\fignote{Figure: pipeline overview diagram. Left-to-right: (a) RTI+photogrammetry capture $\rightarrow$ colour/normal tiles; (b) stitch-type crops $\rightarrow$ stitch LoRAs (one per stitch type, transferable across models); (c) damaged or masked piece + LoRA $\rightarrow$ inpainted fill; (d) fine-tuned Marigold $\rightarrow$ fill normals $\rightarrow$ descriptors $\rightarrow$ Mahalanobis distance to the real type cloud. TO MAKE.}

The workflow has three stages: encoding stitch structure as LoRAs (\ref{sec:method-priors}), reintegrating losses (\ref{sec:method-reintegration}), and verifying fills (\ref{sec:method-verification}). Each stage is deliberately modular: the LoRAs are model-agnostic adapters, the reintegration step uses whatever inpainting interface a given model exposes, and the verification protocol sees only the final RGB fill.

\subsection{Stitch LoRAs and cross-model transfer}
\label{sec:method-priors}

For each stitch type we train a low-rank adapter (LoRA) (rank 16, $\alpha=16$, learning rate $10^{-4}$, batch size 1, 7,000 steps at $1024^2$) on the captioned crops. The same dataset and recipe were applied across four base models chosen to span contemporary architectures and parameter scales: SDXL (UNet, 2.6B, 2023), FLUX.1-dev (rectified-flow DiT, 12B, 2024), Qwen-Image (MMDiT, 20B, 2025), and Z-Image base (single-stream DiT, 6B, 2025). Training the 20B Qwen-Image on a single 32 GB consumer GPU (Nvidia 5090) required weight quantisation during training (3-bit transformer weights with an accuracy-recovery adapter and an 8-bit float text encoder \TODO{cite ai-toolkit / accuracy-recovery adapters}); at inference the same model runs under 8-bit float weight-only quantisation. Further quantisation is possible to accommodate GPUs with lower VRAM.

The key property we exploit is that an inpainting or instruction-edit variant of a base model is itself a fine-tune sharing the base's attention and MLP layers --- exactly the layers a LoRA modifies. A LoRA trained once on the base therefore transfers without retraining onto the variant actually used for reintegration: SDXL $\rightarrow$ SDXL-inpainting, FLUX.1-dev $\rightarrow$ FLUX.1-Fill-dev, and Qwen-Image $\rightarrow$ Qwen-Image-Edit. Z-Image, whose only released variant is step-distilled, is used through its base model; the results demonstrated here strongly suggest that the already trained LoRAs will perform well on the edit variant once it is released (see Section~\ref{sec:results}). This demonstrates that the workflow is generalisable across model architecture, and is robust in the face of rapid development and stable across hardware demands. Just as importantly, the LoRAs are cheap to re-create: a full training run completes in hours on the same consumer GPU, so the workflow does not ultimately depend on any one adapter or any one model when a new family arrives, the same crops, captions, and hyperparameters train a fresh stitch LoRA on it, as demonstrated on multiple open-weight models released over the duration of this project.

\subsection{Loss reintegration}
\label{sec:method-reintegration}

Given a damaged piece, a loss is delineated by a binary mask. In practice this may come from manual annotation or a segmentation tool. In our quantitative experiments synthetic loss is denoted in batches using centred squares of varying sizes. The masked image is then inpainted with the stitch LoRA loaded and a minimal texture-focused prompt containing only the trigger token (e.g.\ ``\texttt{embfnchknt} texture''). For the naive (no LoRA) baseline, the trigger is replaced by a plain-English description of the stitch (``french knot embroidery texture'', ``satin stitch embroidery texture'', ``silk purl embroidery texture''). Prompt wording matters more than might be expected: figurative clauses (``17th-century English embroidery'') cause weakly image-conditioned models to paint a literal miniature picture inside the loss, while dedicated inpainting variants are strongly image-conditioned and resist this; restricting the prompt to texture vocabulary avoids the failure on every variant. We exercise both reintegration interfaces that contemporary models offer:

\begin{itemize}
\item \textbf{Mask-conditioned inpainting} (SDXL-inpainting, FLUX.1-Fill-dev, and masked image-to-image on base models): the mask is an explicit input and the model regenerates only the masked region. Dedicated inpainting variants are strongly image-conditioned and blend fills seamlessly into surrounding context.
\item \textbf{Instruction-driven editing} (Qwen-Image-Edit): the model receives a damaged image and a natural-language instruction (``fill the missing area with \texttt{embfnchknt}...''). This interface extends naturally to whole-object reintegration where delineating individual losses is impractical (Section~\ref{sec:results-real}).
\end{itemize}

The LoRA strength $s \in [0,1]$ scales the LoRA's contribution at inference and gives the workflow a built-in, perfectly controlled baseline: $s{=}0$ is the identical model, seed, and mask, with the trigger token substituted for the plain-English description above. All quantitative comparisons below are of this off/on (and graded-strength) form, so improvements are attributable to the LoRA rather than to model or prompt differences.

%\textbf{Resolution discipline.} Fills are generated at the native resolution of their context: every evaluation tile is a $1024^2$ crop taken directly from the 42 MP capture, and is never rescaled on its way into or out of a model. This matters twice over. Generatively, inpainting into upscaled context produces structured artefacts --- most strikingly, one model's prior collapses the unconstrained interior of a large hole to a centred ``medallion'' motif that no adapter strength can repair \TODO{decide whether to name FLUX.1-Fill here or keep generic; candidate supplementary figure}. Metrically, three of the surface descriptors (dominant pitch, bump density, bump spacing) are absolute-pixel quantities, so a fill is only comparable to a reference computed at the same pixel density; all fills, reference crops, and estimated normals therefore share one native resolution.

%Every model variant, including the 20B Qwen pair under fp8 weight-only quantisation, ran on the same single 32 GB consumer GPU (Nvidia RTX 5090). The full quantitative sweep completed in ~40 hours, ranging from 8 to 90 seconds depending on model size.

\subsection{Verification: from fill to craft fidelity}
\label{sec:method-verification}

\subsubsection{A calibrated normal estimator}
\label{sec:method-marigold}

Verification begins by lifting each fill from RGB into surface-normal space using Marigold-Normals fine-tuned on our corpus. We warm-start from the public \texttt{marigold-normals-v1-1} checkpoint and fine-tune on the 11,813 colour--normal training tiles of Section~\ref{sec:data} for 4,500 iterations at an effective batch size of 16, with a learning rate of $3\times10^{-5}$ under a 100-iteration warmup and exponential decay, horizontal-flip augmentation, and an MSE objective; inference uses 4 denoising steps at a processing resolution of 768 px. Against held-out RTI ground truth the fine-tune roughly halves mean angular error on high-relief tiles (e.g.\ 27\degr$\rightarrow$13\degr; Table~\ref{tab:marigold}, Figure~\ref{fig:marigold-strip}), with the largest gains exactly stitch structure is strongest. This validation step is what licenses the estimator's use downstream as a measurement instrument on generated imagery, where no RTI ground truth can exist.

\begin{figure}[tb]
\centering
\includegraphics[width=\linewidth]{figures/marigold_1.png}
\includegraphics[width=\linewidth]{figures/marigold_2.png}
\includegraphics[width=\linewidth]{figures/marigold_3.png}
\caption{Normal estimation on a held-out tile: colour input, RTI+photogrammetry ground truth, public Marigold-Normals baseline, and our embroidery fine-tune, annotated with mean angular error. Displayed samples are from the 100 highest tiles with largest variance in normal angular direction.}
\label{fig:marigold-strip}
\end{figure}

\begin{table}[tb]
\centering
\caption{Normal-estimation accuracy on held-out RTI tiles: mean angular error (\degr) of the public Marigold-Normals baseline vs.\ our embroidery fine-tune. \TODO{Fill from evaluate\_high\_variance\_tiles.py output; report per stitch type or per variance band.}}
\label{tab:marigold}
\begin{tabular}{lcc}
\toprule
Tile set & Baseline & Fine-tuned \\
\midrule
High-relief tiles & \TODO{27.x} & \TODO{13.x} \\
All tiles & \TODO{} & \TODO{} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Quantitative metric evaluation}

To derive a quantitative metric to evaluate the performance of our generative models, a scoring system that evaluates similarity and feasibility of the generated stitches needs to avoid the dual pitfalls of circularity: evaluating against the same criteria on which the LoRA trains, rather than the properties important to conservation, and of overprescription: evaluating against an exact replication of the original image from which the loss was generated. The conservation question is whether the fill is a feasible piece of the prescribed craft, not whether it is close to an exact replication. We therefore measure fill fidelity with an arrangement-invariant statistic.

\subsubsection{Classical surface descriptors and the type cloud}
\label{sec:method-descriptors}

From the estimated normals of a region we compute a seven-dimensional descriptor: mean and standard deviation of zenith slope; RMS roughness; bump density and median bump spacing (from curvature/divergence blob detection); structure-tensor directional coherence; and dominant spatial pitch from the radial power spectrum. These are deliberately classical, training-free measurements --- placing a learned classifier here would stack a second black box on top of the normal estimator and invite circularity; the descriptors keep the metric interpretable.

Computing descriptors over the real stitch-type crops yields one ``type cloud'' (mean $\boldsymbol{\mu}_t$, covariance $\boldsymbol{\Sigma}_t$) per stitch type. The space is strongly discriminative on real data: nearest-centroid classification separates the three types at 95.4\% (and 89\% when restricted to windows as small as the smallest loss we evaluate \TODO{re-validate at the final window sizes; previous figure was at a 170 px window}), with inter-centroid separations of 5--11$\sigma$. \fignote{Figure: PCA of the descriptor space showing the three real-stitch clusters — validate\_pca (windowed + full-tile versions regenerate with the running sweep's scoring stage; old copies were cleared).}

\textbf{Fill fidelity} is then the Mahalanobis distance from a fill's descriptor vector (computed over the masked region only) to the prescribed type's cloud:
\begin{equation}
D(\mathrm{fill}, t) \;=\; \sqrt{(\mathbf{x} - \boldsymbol{\mu}_t)^{\!\top} \boldsymbol{\Sigma}_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)} .
\label{eq:maha}
\end{equation}
Real held-out crops of the same type sit at a self-distance of $D \approx 2.6$ in this space (the expected radius of the cloud itself), which provides the natural reference: a fill with $D$ near this ring is statistically indistinguishable from real work of that type using our descriptors.

%Two design points deserve emphasis. First, the reference is the type cloud, not the fill's surrounding context. Context-matching measures blending, which every competent inpainter already does well, and actively penalises the valuable case of reinstating a stitch type into a loss whose surroundings are a different stitch. Second, the descriptors include scale-dependent terms (pitch, bump spacing), so each fill is scored against a reference cloud rebuilt from random crops of the real data at the fill's own window size; scoring small windows against whole-tile statistics systematically inflates distances.

% =========================================================================
\section{Experimental Setup}
\label{sec:setup}

\textbf{Quantitative protocol (synthetic losses).} The full factorial sweep crosses the 96 held-out tiles (32 per stitch type) $\times$ 3 loss sizes $\times$ LoRA strengths $\times$ 6 model variants (FLUX.1-Fill; SDXL-inpainting and SDXL masked img2img; Qwen-Image masked and Qwen-Image-Edit instruction-driven; Z-Image). Each variant swept five strengths $s \in \{0, 0.25, 0.5, 0.75, 1\}$giving $96 \times 3 \times \times 6 \times 5 = 8,640$ fills, all at native $1024^2$ with fixed seed, prompt, and mask within each comparison. Every fill is scored by Eq.~\ref{eq:maha} against the size-matched cloud of its prescribed stitch type, and the three stitch types are reported on an equal footing; where a single type is needed to illustrate a mechanism we use French knots, whose discrete, countable relief makes both success and failure easiest to see.

\textbf{Qualitative protocol (real damage).} {TODO - This} Genuinely damaged and unfinished pieces from the corpus are reintegrated with the same LoRAs, using mask-based inpainting for delineated losses and instruction-driven editing for whole-object treatment. No ground truth exists here by definition; these results are assessed visually and by consistency of the fill's estimated normals with surrounding intact work of the same stitch to demonsrate a feasible and worthwhile application.

% =========================================================================
\section{Results}
\label{sec:results}

\subsection{The stitch LoRAs transfer across models}

\begin{figure}[tb]
\centering
\includegraphics[width=.19\linewidth]{figures/french_knot.png}\hfill
\includegraphics[width=.19\linewidth]{sdxl_french_knot_7000.jpg}\hfill
\includegraphics[width=.19\linewidth]{flux_1_french_knot_6500.jpg}\hfill
\includegraphics[width=.19\linewidth]{qwen_french_knot_6500.jpg}\hfill
\includegraphics[width=.19\linewidth]{zimage_french_knot_7000.jpg}\\[2pt]
% \includegraphics[width=.19\linewidth]{sdxl_satin_7000.jpg}\hfill
% \includegraphics[width=.19\linewidth]{flux_1_satin_7000.jpg}\hfill
% \includegraphics[width=.19\linewidth]{qwen_satin_7000.jpg}\hfill
% \includegraphics[width=.19\linewidth]{zimage_satin_7000.jpg}
\includegraphics[width=.19\linewidth]{figures/satin.png}\hfill
\includegraphics[width=.19\linewidth]{figures/sdxl_satin_7000_0.jpg}\hfill
\includegraphics[width=.19\linewidth]{figures/flux_1_satin_7000_0.jpg}\hfill
\includegraphics[width=.19\linewidth]{figures/qwen_satin_7000_0.jpg}\hfill
\includegraphics[width=.19\linewidth]{figures/zimage_satin_7000_0.jpg}
\includegraphics[width=.19\linewidth]{figures/silk_purl.png}\hfill
\includegraphics[width=.19\linewidth]{figures/sdxl_silk_purl_7000.jpg}\hfill
\includegraphics[width=.19\linewidth]{figures/flux_1_silk_purl_7000.png}\hfill
\includegraphics[width=.19\linewidth]{figures/qwen_silk_purl_7000.jpg}\hfill
\includegraphics[width=.19\linewidth]{figures/zimage_silk_purl_7000.jpg}
\caption{The same stitch vocabulary learned on four models from one dataset and recipe. The same prompt and seed is used for each image, keeping a consistent abstract composition in columns 2-5. Columns: Ground truth photograph, SDXL, FLUX.1-dev, Qwen-Image, Z-Image; rows: French knot, satin, silk-purl.}
\label{fig:lora-grid}
\end{figure}

All four base models learned all three stitch vocabularies from the same small crop datasets (Figure~\ref{fig:lora-grid}), and every adapter transferred onto its model's inpainting/edit variant if available without retraining.

\subsection{Stitch LoRAs keep fills statistically real as losses grow}
\label{sec:results-quant}

\begin{figure}[tb]
\centering
\includegraphics[width=\linewidth]{figures/summary_lora_improvement.png}
\caption{\PRELIM{} Mahalanobis distance to the real cloud of the prescribed stitch type per model variant at each loss size, LoRA off ($s{=}0$, grey) vs LoRA on ($s{=}1$, green); lower is closer to real work. \TODO{I don't like this graph as much as the next one with circles - says pretty much the same thing and is a bit ugly and hard to read?}}
\label{fig:headline}
\end{figure}

Table~\ref{tab:headline} and Figure~\ref{fig:headline} give the central result, here illustrated on French knots. \TODO{ALL NUMBERS in this subsection are PRELIMINARY, from the earlier 9-tile French-knot-only run; replace with the final all-stitch native-1024 run, and add the satin/silk-purl results.} Pooled across the six model variants, fills generated without the LoRA ($s{=}0$) drift away from the real-stitch cloud as the loss grows: mean distance $2.07 \rightarrow 2.79 \rightarrow 4.72$ across the three loss sizes. Fills generated with the LoRA ($s{=}1$) remain pinned near the real self-distance ring ($\approx$2.6) at every scale: $1.79 \rightarrow 2.15 \rightarrow 2.37$. The LoRA's benefit therefore scales with the size of the loss: a +14\% mean improvement at the smallest loss (where surviving context lets any inpainting variant perform reasonably), +23\% at the middle size, and +50\% at the largest, where the surviving context offers least guidance. At the largest loss every individual model variant improves with the LoRA on; at the smaller sizes five of six do, with the exceptions marginal \TODO{re-identify the exceptions on the final run}.

\begin{table}[tb]
\centering
\caption{\PRELIM{} Mahalanobis distance to the real French-knot cloud (lower is closer to real; real held-out crops self-score $\approx$2.6), pooled over six model variants. \TODO{Numbers from the earlier French-knot-only run; replace from the final all-stitch run and add per-model / per-stitch tables or appendix.}}
\label{tab:headline}
\begin{tabular}{lccc}
\toprule
& \multicolumn{3}{c}{Loss size (px on $1024^2$ tile)} \\
\cmidrule(lr){2-4}
& 128 & 256 & 512 \\
\midrule
LoRA off ($s{=}0$) & 2.07 & 2.79 & 4.72 \\
LoRA on ($s{=}1$) & \textbf{1.79} & \textbf{2.15} & \textbf{2.37} \\
\midrule
Mean improvement & +14\% & +23\% & +50\% \\
Variants improved & 5/6 & 5/6 & 6/6 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[tb]
\centering
\includegraphics[width=\linewidth]{figures/cloud_map_sm_pooled.png}
\caption{The result in the metric's own space: whitened descriptor-space panels per loss size, pooled over models. LoRA-off fills (vermillion circles) scatter away from the real cloud as the loss grows; LoRA-on fills (blue crosses) stay inside it. \TODO{Make this all a bit bigger}}
\label{fig:cloudmap}
\end{figure}

The descriptor-space view (Figure~\ref{fig:cloudmap}) makes the mechanism legible: at the largest loss, LoRA-off fills leave the real French-knot distribution entirely (typically toward smoother, lower-relief, less-bumped surfaces --- plausible cloth, but not French knots \TODO{verify the direction of failure against per-descriptor z-scores}), while LoRA-on fills remain inside the cloud of real work.

\subsection{Strength response and the built-in baseline}

\begin{figure}[tb]
\centering
% \includegraphics[width=\linewidth]{summary_strength_sweep_sm.png}  % <- uncomment when the running sweep regenerates it
\fbox{\parbox{0.9\linewidth}{\centering\vspace{2em}\TODO{figure lands from running sweep: summary\_strength\_sweep\_sm.png}\vspace{2em}}}
\caption{\PRELIM{} Distance to the real cloud vs.\ LoRA strength per model variant at each loss size.}
\label{fig:strength}
\end{figure}

At the largest loss size the response to LoRA strength is monotonic for all variants, confirming that the measured gain tracks the LoRA's actual contribution rather than incidental pipeline differences (Figure~\ref{fig:strength}). At small ($128{\times}128$) losses the curve is flat: with abundant surviving context, the fills are near-real with or without the LoRA, and there is simply little reintegration left to do, so models with no LoRA can often perform the task adequately.

\subsection{Reinstating a stitch against a different surround}
\label{sec:results-switching}

\TODO{Figure: the heterogeneous ``switching'' case a loss crossing a French-knot feature embedded in a satin ground (or similar): LoRA-off fill bleeds the surrounding texture into the loss; LoRA-on fill reinstates the prescribed stitch. Before / LoRA-off / LoRA-on / normals. TO ASSEMBLE.}

Real embroideries are not uniform fields of one stitch, and the most consequential losses interrupt a feature --- a line of knots crossing a satin ground, a purl-work border against couched silk. This case is the motivation for the metric design: an unconditioned inpainter, by construction, continues the surrounding texture into the loss, and any context-matching score would reward it for doing so; only a LoRA that encodes the prescribed stitch can reinstate it against the pull of the surrounding visual context. \TODO{Describe the demonstration: which piece/feature, which models, visual result; - remove some of the french knot "clouds" entirely, maybe a satin bird, a patch of silk purl?}

\subsection{Real losses: qualitative reintegration}
\label{sec:results-real}

\fignote{Figure: real damaged/unfinished pieces reintegrated — before / masked / after / fill normals. hair / castle / snail?}

\TODO{Describe demonstrations}

% =========================================================================
\section{Discussion}
\label{sec:discussion}

\textbf{Conservation framing.} The workflow is an instrument for conservation, where digital fills cost the object nothing and are reversible by deletion; they should always be presented as distinguishable interpretations, a digital analogue of tratteggio, and the verification protocol exists precisely so that the decision to show a fill can rest on evidence rather than plausibility. We see the immediate uses as treatment planning (visualising candidate reintegrations before any needle touches thread), interpretation of unfinished or damaged works, and engagement applications

\textbf{Why model-independence matters.} During this project's lifetime, multiple base models in our original set were superseded and the successors integrated into the workflow. Any workflow whose validity is bound to one checkpoint is bound to be usurped in a matter of months. Ours binds the heritage-specific assets: captured corpus, stitch LoRAs, verification protocol, to the stable interfaces (low-rank adaptation, mask- or instruction-conditioned generation), and we demonstrated the same LoRAs and the same metric operating across six inpainting/edit variants of four model families.% (with priors additionally trained on a fifth \TODO{FLUX.2-Klein — decide whether it joins the sweep or is mentioned as priors-only}). The expensive, object-derived components survive model churn.

\textbf{An objective inpainting evaluator.} Beyond this application, the verification chain is a contribution in its own right: an objective, automatic evaluator of inpainting quality for relief surfaces, replacing the subjective visual judgement on which generative restoration work usually rests. Its objectivity is anchored at both ends. At the sensor end, the fine-tuned estimator is validated against physical ground truth on the same corpus before it is ever pointed at generated imagery. Downstream of the sensor, everything is classical and transparent: seven interpretable measurements and a covariance-weighted distance, with no learned judge whose own biases would need auditing, and every score decomposable into per-descriptor terms (``real French-knot work has bump density $\mu \pm \sigma$; this fill sits $z$ standard deviations away''). The same construction applies to any craft surface for which physical ground truth can be captured, and we expect it to be useful beyond embroidery. The residual risk is that the estimator hallucinates plausible normals on generated textures unlike anything it was trained on; this is bounded by the fact that it was trained only on real embroidery, so systematic flattery of generative artefacts is unlikely.

\textbf{Limitations.} The quantitative result covers three stitch types of tiles drawn from a small number of pieces \TODO{rephrase once the all-stitch run lands; per-type tile counts}. Square synthetic masks are a convenient proxy but real losses have irregular boundaries and degraded margins. The metric is geometric only: colour and material fidelity (thread lustre, metal vs. silk) are currently judged by eye. Scale must be handled explicitly. The descriptors are scale-dependent and the reference cloud must be window-matched, which makes cross-resolution comparison delicate. Finally, fills at loss sizes beyond those tested remain unquantified; we expect a fidelity cliff somewhere beyond this and locating it matters for practice.

\textbf{Future work.} The most direct extension is breadth: growing the three stitch vocabularies into a proper corpus of thread and technique. Building a library of more stitch types, more thread materials and colourways, crops drawn from more pieces and periods so that a conservator can search, request, or upload any technique present in the work under treatment, in the form of a LoRA. Second, the descriptor set should grow beyond geometry: colour and material descriptors (thread lustre, metallic versus silk reflectance, dye palette) would bring the dimensions currently judged by eye under the same objective protocol. Such LoRAs can be combinatory up to a point, such that lustre, techique, and colour may be separately trained LoRAs that may be combined. Third, the evaluator itself needs human anchoring: a study with practising conservators and embroiderers, testing both whether LoRA-conditioned fills are judged craft-plausible and whether the descriptor distance agrees with expert assessment of feasibility. Finally, verified normal maps are precisely the input that relief-printing and tactile-reproduction pipelines consume, which closes a satisfying loop: a fill that is correct in normal space can be not merely displayed but fabricated, returning a touchable, intact surrogate of a damaged object to audiences who may never have the opportunity to interact so closely with the original.

% =========================================================================
\section{Conclusion}
\label{sec:conclusion}

We present a model-independent workflow for digitally reintegrating losses in heritage embroidery: stitch LoRAs trained from one dataset and recipe on four generative families and deployed on each family's inpainting or edit variant; loss reintegration through whichever mask- or instruction-conditioned interface a model exposes; and verification in surface-normal space, where an RTI-validated estimator and a classical descriptor metric measure whether a fill reproduces the prescribed craft structure rather than merely looking plausible. On ground-truthed synthetic losses, unconditioned inpainting drifts away from real stitch statistics as losses grow, while LoRA-conditioned fills remain statistically indistinguishable from real work at every scale tested, across every non-distilled model variant evaluated. The components that embody the heritage object (corpus, LoRAs, estimator, protocol) are exactly the components that survive the churn of generative models, which is exactly what a workflow for slow-moving, delicate collections requires.

%%% --- bibliography ---
\bibliographystyle{eg-alpha-doi}
\bibliography{references}



\end{document}
