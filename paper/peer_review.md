# Editor comments
## Originality (Novelty): 1

As the authors suggest in their notes, this is the among the first implementations of this open source code in this context, and the ability to evaluate uncertainties is novel. Gravity inversions are, of course, relatively routine for sub-ice shelf bathymetry but the implications for future survey priorities are well demonstrated here.

## Scientific Quality (Rigour): 1

A comprehensive set of tests is undertaken: I wonder if reviewers will request some abbreviation of the testing approach, but the rigour is certainly there in the first instance.

## Significance (Impact): 1-2

Other than demonstrating the use of a new open source model, the significant implication from this work is how to best improve estimates of sub-shelf bathymetry. What is less clear is how much practical difference (e.g., to sub-shelf ocean circulation models) such insight will offer, but the ability to identify these priorities in the first place is clearly advantageous.

## Presentation Quality: 2

The text is quite dense in places, and there are a lot of figures - some of which (e.g., Figure 16) are quite difficult to interpret. However, I can see that there are some complex interactions to try to convey and so I am satisfied to await the views of reviewers. I also wonder if some thought could be given to reducing the number of figures.

# Reviewer 1
 > Thank you very much for your insightful comments and taking time to review this paper! We agree with your major point about the gravity data noise value, and have re-run all relevant portions of the code used a large amount of noise. This has highlighted gravity noise as a more significant factor in bathymetry inversions than the original manuscript suggested, and we will re-word some of our discussion and conclusion remarks to include this. We have responded inline to each of your comments below with indents.

## General comments

This paper aims to provide a robust theoretical background and test for practicality and usefulness of gravity inversion for determining sub-ice shelf bathymetry. Such a paper is useful as it has the potential to guide and optimise future real-world data collection over Antarctic ice shelves. The paper uses a prism-based forward model, coupled with an iterative least-squares approach to provide bathymetric estimates. The test results point towards the importance of higher quality/resolution gravity data in areas of low amplitude background field (simple underlying geology), while direct observations (e.g. seismic or AUVs) become increasingly important where the underlying geology is complex.

Overall the paper is well written and the results appear reasonable. However, I have one specific comment associated with the treatment of gravity errors which I feel should be addressed and a few additional more technical points. This will likely not significantly change the outcome of the paper, but may change the suggested likely minimum achievable error in bathymetry from gravity data.

## Specific comments

Around L390 to 395 the authors talk about simulating noise in the gravity data. My understanding of the paper is that the authors simulate noise by first adding random Gaussian noise to the baseline gravity disturbance. This pixel by pixel noise has an amplitude in-line with the errors reported for typical airborne surveys. The initially adulterated data is then re-filtered to achieve a best noise reduction with minimal loss of gravity signal, and the subsequent re-filtered data inverted for bathymetry. However, the data loss from noise and filtering Fig. 10c is consistently below +/-1 mGal, which seems small compared to what would be expected for a real survey.

The authors justify re-filtering the data after adding noise because filtering is a standard method of noise reduction in airborne gravity processing. However, the errors quoted for gravity surveys are after filtering. I therefore don’t think this is the best way to simulate noise in a synthetic gravity dataset. I would suggest that a better method would be to create a random Gaussian noise field, which when filtered with a 10 km wavelength filter (to simulate gravity processing) had a 1 mGal standard deviation (equivalent to the error in high quality gravity data). Adding this filtered error field (with likely local maximum amplitudes of +/- 4 mGal) to the baseline gravity disturbance would be more representative of the likely errors in real Antarctic airborne gravity data. Other ways to create realistic noise could be considered. Use of this error field would likely amplify the errors in the recovered bathymetry, giving a higher, but more realistic, estimation of the expected error due to noise in the gravity data.
 > This is a good point, thanks for pointing it out. We agree the tested noise value was at the optimal end of realistic noise levels. We have re-ran Tests 2 and 5 using pseudo-random noise with a standard deviation of 3 mGals. This resulting in peak errors of ~12 mGal. Filtering with a 10km Gaussian filter width gave RMSE of ~1 mGal. We have retained the trials of different filtering wavelengths, since we believe these results in themselves are informative. The optimal filtering results was found to be ~24 km, which is generally larger than we would have expected. This reduced noise RMSE to ~0.6 mGal. Additionally, for Ensembles 2-4, we have changed the tested noise limits (pre-filtering) from 0-3 mGals to 0-5 mGals. We will revise the manuscript to reflect these results using the more realistic noise levels. This will result in updates to Figures 8, 9, 10, 13, 14, 15, and 17, as well as all relevant portions of the text.

## Technical corrections

L35 and other places in the text (e.g. L277, L343) refer to “regional gravity field strength”. It is not 100% clear what is meant by this. My understanding of this in other contexts in the paper is that the authors mean the “amplitude of the variability in the regional field”. High field strength could be a uniform value of 200 mGal, but this would have no impact on the inversion quality.  I would suggest re-wording.
 > Yes you are correct when we refer to `strength` it is the amplitude of the variability, aka the standard deviation. Thank you for point out this confusion. We will update the text to be clearer, for example changing "weak regional fields" to "regional fields with low-variability".

L163 – It is not clear why the sensitivity matrix is populated by the vertical derivative of the gravity. This should probably be justified in a little bit more detail. – I think high gradient areas might have shallower sources so be more sensitive, but this is a guess? This is covered in Appendix 1, which could be cited. However, in the appendix the example of varying density was given. As this is fixed in the inversion then the matrix can be filled just with the gravity gradient. However, the parameter which is varied is the topography, which isn’t fixed at each iteration. Therefore is the sensitivity matrix re-computed at each step as well (L191-193)?
 > Thanks for pointing out this mistake. We will update the text to read "Each entry in the Jacobian is the partial derivative of gravity produced by a specific prism at a given observation point, with respect to the prism's thickness." I think `partial` as accidentally replaced by `vertical`. And yes because the Jacobian depends on the current model of bathymetry, it has to be recalculated at each iteration.

L210 – constructing training datasets for Damping value cross-validation. This is done by creating two raster’s – training and testing, which are on meshes with cell size X, shifted by ½ X. In effect taking a mesh with cell size ½ X and considering alternating points.  A concern with this is that the mesh size X must leave some ambiguity. For example if you have 10 km wavelength gravity data and training/testing meshes of 100 m both will be in effect identical. Mesh size therefore matters in this case and is related to the wavelengths considered. The mesh size used for generating the observation and test data, or how it could be estimated, should be stated here.
 > This is not exactly how the training and testing set are created. If the original gravity data was on a 10 km grid, then it is interpolated onto a 5 km grid. The points which fall on the original 10 km grid now make the training set, and the new (interpolated) points which fall between the 10 km grid points are the testing set. With this technique, we don't require an arbitrary mesh size, we just use half of the original cell size. There could be some testing for whether 1/2, 1/4, or 1/8 the mesh size is optimal, but our results (and those of Uieda et al. 2016) appear reasonable, and we think this is out of the scope of this work.

229-237 – Uncertainty constraint. It is not clear if/how the uncertainty is quantified given the control points form part of the inversion, so should have zero offset. Were random control points left out?
 > We didn't include the depth uncertainty of the constraint points in our uncertainty analyses as there were already many components in our uncertainty analysis.


# Reviewer 2
 > Thank you reviewer 2 for your time and efforts to review this manuscript! We think your suggestions have improved this work and we have responded inline to each of your comments below with indents.

This manuscript presents a new open-source geometry inversion tool (Invert4Geom) for recovering sub-ice-shelf bathymetry from gravity data, together with a comprehensive suite of synthetic tests and Antarctic ice-shelf survey analysis. The authors demonstrate the algorithm’s behavior under ideal and realistic conditions, investigate the influence of key parameters (data noise, survey spacing, regional field strength), and quantify uncertainty via Monte Carlo sampling. They conclude with practical recommendations for future airborne gravity surveys and bathymetric constraint collection.

The topic is timely and the open-source implementation will benefit the glaciological and geophysical communities. The manuscript is generally well structured, and the figures are clear. However, some areas require clarification or rephrasing to improve readability and scientific rigor. In particular, I list some minor comments to help strengthen the manuscript.

L7 & L24 & L293 Definition of “real” vs “synthetic” bathymetric data
 > We will replace "real" with "high-resolution multibeam" to be clearer.

L313 The description of the four ensembles (especially the parameter ranges and sampling strategy) remains too general.
 > We will add additional descriptions to each of the bullet points for the ensembles, describing them in more detail.

Figure 8 The thick grey line in the profile panels can be misinterpreted as an uncertainty envelope. Replace the thick grey line with a thin black line for the profile of the inverted bathymetry, and show the starting model with a dashed line.
 > Good point, we will update this.

Figure 12 Use a slightly darker color for the “true regional” field so it is distinguishable from the estimated field.
 > Yes will do.

L524 You introduce “RMSE” and then immediately write it out (“root mean squared error”). This is redundant.
 > Fixed.

Figure 16 The red and black colors in the ice-shelf names denote previous versus new inversions, but this could be repeated in the figure caption for clarity.
 > Added.

Section 3.9 presents results that the authors acknowledge are “expected. You could move the detailed maps and synthetic summaries of Section 3.9 to the Appendix, and condense the main text to a short paragraph highlighting only the key findings.
 > While the results may have been expected, we think they are still informative and worth including in the main text, but agree that this could be a good place to remove some length. We will move the central column of Figure 17 (Ensemble 3) to the supplement, and consolidate the text to be more brief.

L590 Change to “resembles those of Ensemble 2”.
 > We will fix this.

L675 Do you mean “dense constraints”?
 > No, we are referred to the general poor performance of inversions where there is a strong regional field a few (sparse) constraint points.

L683  Change “Gravity inversions in Antarctica…” to “Gravity‐based bathymetry inversions...”
 > Here we are referred to the fact that topography/bathymetry derived from gravity inversions is currently constrained to sub-ice shelves, but we are suggesting this method may work for non-bathymetry purposed, i.e. grounded bed topography.

Recommendation:

Once revised, this work will be a valuable resource for the bathymetry‐ and ice-sheet modeling community.


# Reviewer 3
> Thank you reviewer 3 for your thorough review! We really appreciated the time and detail you have put it, you comments had greatly improved the manuscript, and help point out locations where we were unclear. We think there was a little misunderstand on how AntGG was included in our work, and hopefully we can revise the text to limit this for other readers. We have responded inline to each of your comments below with indents.

## General comments

This paper employs a rigorous method to test gravity-based bathymetric inversions for sub–ice shelf cavities using a new open-source algorithm. The algorithm calculates gravity reduction through a forward model that uses prism-based density discretization. It assesses gravity misfit and integrates bathymetric constraint points to minimize the influence of regional gravity effects on the inverted bathymetry.

Synthetic tests were conducted in front of the Ross Ice Shelf to estimate the influence of gravity noise, gravity measurement spacing, and long-wavelength gravity distribution on inversion uncertainties. These uncertainties are computed via Monte Carlo simulations. This work has the potential to significantly impact the scientific community, as gravity-based bathymetric inversion remains one of the few viable techniques for estimating bathymetry beneath ice shelves in the absence of direct measurements.

Although various techniques have been developed and applied, quantifying their associated uncertainties and determining how closely they reflect actual bathymetry remain challenges. The tools proposed here represent a valuable and timely contribution to addressing this gap.

The tests are well designed, and the figures are clear. The paper uses several methods for estimating uncertainties with optimized parameters, presenting a structured and coherent approach focused on the primary objective.

## Specific Comments

You need to be careful with your choice of test region. You selected the area in front of the Ross Ice Shelf due to its dense bathymetric coverage. However, (1) it does not cover a sub–ice shelf cavity area, and (2) it is a region with almost no gravity measurements in the ANTGG2021 dataset (see ANTGG2021's Standard Deviation map in Supplement).
 > Yes the choice of study region was difficult. We required somewhere that we thought would generally reflect sub-ice shelf bathymetry, but had high-resolution bathymetry data and basement topography (for the regional component of gravity). However, since our tests used synthetic gravity data, from the forward modelling of the bathymetry and basement topography, we didn't actually use the gravity measurements from ANTGG2021. We thought this distinction between real and synthetic gravity data was clear, but we will revise the relevant text to make it more clear.

Since your aim is to estimate uncertainties for sub–ice shelf cavities, it would be more appropriate to perform your tests in a region that actually includes a sub–ice shelf cavity. While it is difficult to find ice shelves with dense bathymetric coverage, some areas do have a substantial number of gravity measurements. You could consider comparing estimated uncertainties from tests conducted over an ice shelf with those from open-ocean regions where both gravity and bathymetry are well constrained.
 > Yes we agree that ideally we would have a densely surveyed sub-ice shelf cavity we could use as a test case, but the most densely surveyed ice shelf (Thwaites) has a median distance to the nearest bathymetry measurements of 2.5 km, significantly larger than the 400 m value for our Ross Sea survey area. So if we used the 2.5km resolution bathymetry as our "truth", we would only be accessing how well the inversion works at recovering features with a 2.5km or greater wavelength.

You should more clearly define the ANTGG2021 gravity grid, as it is a major source of uncertainty in gravity-based bathymetric inversions. The ANTGG gravity grid combines all available gravity measurements (which do not cover the entirety of Antarctica) and estimates gravity in data-sparse regions using satellite observations (GRACE and GOCE) and topographic data from Bedmap2 (see Hirt et al., 2016; Scheinert et al., 2016; Zingerle et al., 2019; Zingerle et al., 2021). The use of Bedmap2 topographic data to reconstruct gravity in areas lacking measurements is especially concerning, since you ran your tests in an area with very limited direct gravity data (see the figure below showing the standard deviation map of the ANTGG2021 grid). As a result, your inverted bathymetry is likely very similar to the measured bathymetry because the gravity signal was reconstructed from Bedmap2 data. This introduces bias into your results. However, if you were to relocate your tests to a region with dense gravity and bathymetry measurements, your results would likely improve significantly and avoid this bias. See attached file with ANTGG2021 grid standard deviation map.
Your goal is to enhance gravity-based bathymetric inversions for sub–ice shelf cavities. While many studies have previously addressed this topic using various methods and assumptions, it is unclear whether the uncertainties and test results in your work are applicable to existing gravity inversion techniques. Adding a section that clearly reviews the current state of the art—including the limitations of previous studies—would fill this gap. A map showing the inverted bathymetry derived from your method, along with the differences compared to prior models, would help identify where your approach is most effective and where existing inversions might need to be recalculated due to high uncertainty or outdated methods.
 > I think there is some misunderstanding of our method, as we have not used any gravity measurements in our inversions, only synthetic gravity calculated from the forward-modelling of bathymetry (and basement) topography. We have used this approach so the only errors are those we have introduced ourselves in the various tests (adding noise, adding regional fields, reducing resolution via airborne surveys etc.). AntGG2021 was only used in the ice shelf analysis to estimate the regional field standard deviation for each ice shelf.

## Technical comments

L14-16 “We analyzed Antarctic ice shelves and found that, if high-resolution gravity data were available, gravity inversion could improve bathymetry models for 94% of them compared to interpolated products like Bedmap2.”  Does that mean that we only have 6% of high-resolution gravity data covering Antarctica's ice shelves?
 > No, we were trying to say that if you were to have high-resolution gravity surveys over all Antarctic ice shelves, and you performed gravity inversions for bathymetry for each shelf, for 6% of the shelves the inversion would results in a worse inversion than the current Bedmap2 model. We will try and reword this to be clearer.

L18-20 To rephrase in a better logical order. (1) The shape and depth elevation of the continental shelf seafloor may allow warm water (name of warm water, usual depth) to reach the subglacial cavity. (2) The shape of the subglacial cavity may lead warm water to reach the grounding zone, where the ice is at its deepest, steepest, most pressurized, thus vulnerable to significant basal melting. (3) This significant basal melting may affect the stable state of ice shelves due to the long-term retreat of their grounding line.
 > Thanks, that is better, we will update the text.

L21 “echo sounders”. What about seismic data?
 > We will add seismics to the sentence, thanks.

L21-22 “ are often impractical or expensive when applied to the vast ice shelves that fringe Antarctica’s ice sheets.” Why? Explain that this is because the instruments have to be sent under the ice using AUVs (Automated Underwater Vehicles).
 > Yes good point, we will re-work this sentence, since "conventional" for ice shelves is actually over-ice seismics which we left out.

L22 “acquiring”. Do you mean measuring or estimate? Acquiring is too vague.
 > See above.

L22 “gravity”. When you say "gravity" are you talking about free-air gravity anomalies?
 > In the rest of the article, gravity observations refer to measurements after field corrections, such as drift, levelling, Eotvos. The anomaly used in the inversion is a derivation of the gravity disturbance, which is largely referred to as the free air anomaly, but technically it is a distinct type of anomaly. Since this is the plain language summary we use gravity to try to avoid jargon.

L23-24 “difference in density between seawater and the seafloor.” What about the ice density?
 > Yes that also produces a gravity effect, which we account for in the gravity reduction procedure, but just didn't want to include it in the plain language summary for simplicity.

L24 “a gravity inversion”. Using which gravity data/model? In which areas?
 > We use purely synthetic gravity data, but to keep the plain language summary short we haven't gone into the details here.

L24 “synthetic data”. To define, you are also using existing datasets to set up your tests, right? Not all data are synthetic.
 > No the gravity data used in all the inversions in this work are synthetic, generated from the forward-modelled gravity effect of topography models. While this use of synthetic gravity data has no practicality in understanding sub-ice shelf bathymetry, we are using it just to explore the sensitivity of the inversion algorithm to various factors, like data noise, spacing, and regional field strength.

L25-26 “We find that removing the portion of the gravity data that results from deep geologic structures is the largest source of error.”  Does that mean that we poorly remove the deep geologic gravity signal? Does this affect our free-air gravitational anomaly signal, which is therefore not correct? The deep geological gravitational signal occurs at long wavelengths and the surface topography gravitational signal occurs at shorter wavelengths. Are you talking about deep geology or bedrock geology (which can vary locally, implying changes in density)?
 > Yes the estimation (and removal) of the deep geologic gravity signal is notoriously difficult. However, this doesn't effect the free-air gravity anomaly, since when calculating the free-air anomaly from the observed gravity you don't remove the regional gravity field. In typically workflows, you only remove the regional gravity just prior to modelling, while analysis of the free-air anomaly is done earlier. There are many ways of defining regional vs residual signals. For us, the residual signal is everything resulting from deviations between the real bathymetry and our starting model. This includes deviations between our assumption of constant density and the true density distribution of the seafloor / underlying Earth.

L30 “existing bathymetry models”. The existing bathymetry models are also gravity inversions. Is your inversion method better than existing ones?
 > We will rephrase this to "would resulting in an improvement bathymetry models obtains from interpolation of point-measurements". And no, we don't specifically argue our inversion is better than others, just that it is open-source, and well-suited to bathymetry inversions, which some other algorithms/software are not.

L42-43 “These ice shelves play a key role in holding back the flow of inland ice by exerting a resistive force, buttressing, which comes from lateral drag and resistive stresses where the ice touches the sea floor at pinning points “. A lot of redundancies can be avoided in this sentence.
 > Ok thanks, we will reword this sentence.

L45  “across the grounding zone”.  I am not sure if it is correct to say that. The grounding zone is changing over time and retreating due to the same processes (not independent of the reducing buttressing effect). You might have wanted to say: "toward the ocean" instead?
 > Good point, yes will change this sentence.

L49  “cold-water shelves”. Why are you talking about the cold-water shelves to highlight the importance of sub-glacial bathymetry? Cold-water shelves are in general in a steady state. It means that they are currently not affected by changes in the ocean currents due to global warming. You should explain instead the unsteady-state ice shelves that are affected by the intrusion of the warm Circumpolar Deep Water reaching the grounding line thanks to the sub-glacial bathymetry.
 > We discuss the cold-water shelves because these shelves tend to experience more Mode 1 melt, concentrated at the vulnerable grounding line. Here, bathymetry is an important factor since it directly controls where the water can reach. Warm-water shelves tend to have a lower percentage of there melt at the grounding zone, and more near the ice front, where bathymetry plays a limited role. We think bathymetry is still an important factor for warm-water shelves, but think understanding grounding-zone melt is where bathymetry products can be most useful.

L49 “grounding zones”. To define before. It could be: The grounding line is the transition point where the ice goes from grounded to floating. This line migrates over the grounding zone due to short-term effects of the tides, and long-term effects of the basal melt or refreeze.
 > Yes we can add a definition of grounding zone here, thanks.

L51 “This water”. Which water are you talking about? "This water" is confusing because you're talking about cold and salty waters, and basal melting (thus fresh water) before, making it confuse.
 > Good point, we will clarify this, but we are talking about the cold salty waters.

L51 “such as”. If you say "such as" it means that the dense and cold water you're talking about can come from a different origin than the one formed from sea ice formation. Explain which one, it should be the water originating from the deep ocean.
 > We are referring the the water from sea ice formation (HSSW) in this example, since it's a clear example of how basalt melt at the grounding zone is controlled by bathymetry troughs.

L57-58 “(AUVs) are impractical for large ice shelves.”  They are not impractical for large ice shelves, because such data were measured on large ice shelves. I guess you wanted to say that it is impractical to cover all the surface of large ice shelves with such measurements because they are point data.
 > Good point, we will rephrase this.

L58 “than water”. To add “and than ice”.
 > Thanks, we will add this,

L59 “gravity”. Do you mean free-air gravity anomaly field?
 > Well it does create a signal in the free-air anomaly, but that is just a specific anomaly type, so more broadly it creates a signal in the gravity field.

L61 “Antarctica”. Cite Charrassin et al., 2024, bathymetric inversions have been calculated for the whole of Antarctica. You cite this article at the end of your paper, but it's important to refer to it first as a state of the art, as it is the most recent gravity-based inversion to have been carried out on such a large scale.
 > Yes good point, we will add this here.

L65 “They’re”. To replace by “They are”.
 > Will change.

L69 “that’s” to replace by “that is”
 > Will change.

L70 “don’t do a good job of” to replace by “struggle to”?
 > Will change.

L71 “geometry inversion”. Why did you choose to do a geometry inversion then? Is it better than density inversions? Why?
 > We require a geometry inversion for this since it is the geometry of the seafloor we want to model. In our case, a density inversion would give us a model of seafloor and subsurface density distributions, which is not what we are interested it. We can add some words to clarify this choice.

L71 “It’s” to replace by “It is”.
 > Will change.

L72 “from”. Add “inverted from”.
 > inversion is the process of modelling topography from gravity, so "to model topography inverted from gravity" is redundant.

L74 “regional gravity fields” to define.
 > Yes good point we haven't defined it yet. We will add this.

L76-77 “Finally, we highlight which ice shelves are most likely to benefit from inversion and which ones probably won’t.” to rephrase maybe like “we highlight ice shelves for which bathymetry would be improved using a free-air gravity anomaly inversion, and ice shelves were it would degrade the bathymetry reliability".
 > Yes that is clearer, thanks!

L80 “forward model” to define.
 > Ok we will move this from the supplement to here.

L107-108  “These sources can then be used to predict the gravity anomaly at any desired point, such as each location on an even grid.” This is not clear. You use the “equivalent sources” technique, which calculates the expected gravity from the observed bathymetry and inserts it into your matrix as input, right? Will this improve the Jacobian solution to be more accurate? If you don't have bathymetry data over a large area, how do you know it really works?
 > Yes the equivalent source technique is essentially a geophysically-informed interpolation method, turning discrete point measurements into a full gridded dataset. If there is a large data gap in the gravity data, the interpolated gravity data in the gap will be smooth. The estimated regional field will them closely mimic this smooth gravity, so residual gravity will be close to 0, and thus the starting model will most remain unchanged in these gravity data gaps.

L109  “)” parenthesis to remove.
 > Thanks.

L121 “densities(Figure 3b)”. Add space.
 > Thanks.

L134 “most of which” to cite.
 > will add reference here.

L147-148 “it avoids the subjective parameter selection required by the other techniques.”. This sentence is too general. The use of constraint points is one of the only known ways of using actual bathymetry measured in inverted results. They are therefore essential for representing bathymetry accurately and avoiding false assumptions about bedrock density. This allows us to avoid using very precise geological assumptions, since they will be corrected thanks to the actual bathymetry data.
 > We will add " required by other techniques, such as low-pass filter widths or upward continuation heights." And yes we agree constraint points are the foundation for any hopes to achieve accuracy in a bathymetry inversion!

L166 “sector of an annulus”. How do you choose the diameter? Is it the initial horizontal length of the prism?
 > Yeah, the diameter ("width" of the annulus) is the same as the prism width.

L175 “to remain close to known bathymetry measurements”. Do you keep the exact values of existing bathymetric measurements in the final inversion results?
 > It's almost impossible to keep them exact, as even the starting model, from the interpolation of the bathymetry points, generally alters the value at the points, unless you use a very restrictive interpolation which then create other artifacts. Our test 5, the semi-realistic scenario, had an RMSE to the constraints of ~12 m. If adhere to the constraints is very important, this RMSE can be reduced to 1-2 m by using a weighting grid during the inversion, but this can add significant time for the inversion to converge.

L204 “Too high a value under-fits the data; too low”. To replace with “Too high,  […] too low, […]”.
 > We will update this sentence.

L255 “it’s” to replace by “it is”.
 > Will do.

L259 “it’s” to replace by “it is”.
 > Will do.

L260 “doesn’t” to replace by “does not”.
 > Will do.

L280-281 “included data from multibeam and single-beam sonar, over-ice seismic surveys, airborne and surface-based radar, and exposed rock outcrops.”. To cite.
 > The citations are on the prior line, Dorschel et al. 2022 and Morlighem, 2022.

L283 “AntGG-2021 gravity data”. You should know that this gravity grid partially uses inverted Bedmap bathymetry to reconstruct gravity when it is not measured. You need to use the error map they have created for your results to be correct.
 > Interesting, thanks for pointing this out. However, we only use AntGG-2021 for estimating the strength of the regional field, and not as input to any inversions.

L283 “Bedmap2”. I think they have added a lot more measurement data (outside the ice shelves) in the Bedmap-3 version. Perhaps it's better to use Bedmap-3 and place Bedmap-2 at the location of the ice shelves? Why don't you want to use the gravimetric inversions that already exist? Your final bathymetry will be different and improved anyway, won't it? Bedmap is just the initial bathymetry to be improved.
 > We aren't actually performing any inversions with this gravity data. We just use it to estimate each shelf's regional gravity field strength, and use that value to predict inversion performance. Even so, if we were performing inversions we would likely avoid using a prior inversion result as the starting model. In most area, the starting model would be changed and it wouldn't matter which starting model you use, but in some area the starting model would remain minimally changed, and therefore the starting model choice is reflected in the final product.

L290 “a range of published sub-ice shelf bathymetry studies”. To cite.
 > Will add these.

L292 “showcase” to replace by “introduce”?
 > Will change.

L339-340 “he standard deviation of the topography-corrected
 > Sorry not sure what this comment means.

L340 “gravity disturbance and the standard deviation of the regional gravity misfit.”. Do they consider the standard deviation of the initial ANTGG grid?
 > No, we don't use standard deviation of the initial AntGG grid (the free-air anomaly) since the values would be increased for shelves with high topography, since the gravity effect of topography is still included in the free-air anomaly. We want to know the standard deviation of the regional gravity field, which by our definition is from all subsurface effects, not topographic effects.

L346 “we reviewed published studies”. To cite.
 > Will add.

L371 “true value”. Cite. If you say it is a true value, it sounds like it has been measured? The word “real” may be a bit strong, perhaps you meant “most realistic”?
 > True refers to the value used to create the synthetic gravity data. We created a prism model, assigned each prism a density, and calculated the gravity effect of those prisms. We then try to recover what density value was original used. We will remove the word true.

L412 “The synthetic airborne survey follows a typical Antarctic design”. To rephrase, what is a “typical Antarctic design”?
 > We will add more detail here. We refer to a grid of perpendicular lines, with spacing ~5-15 km in the main survey direction and a courser spacing, ~50 km, for tie lines.

L437 “regional component”. Maybe you could define again what is it?
 > Yes will add this.

L438 What is “crystalline basement topography”?
 > This refers to the shape of the contact between sedimentary rock and crystalline / metamorphic basement rock. We can clarify this.

L438-439 “Brancolini et al., 1995) ”. Is this citation from the ANTOSTRAT seismic compilation? If so, does this mean that it contains only seismic measurements taken before 1995? Why did you use it when it is not updated?
 > While more data has been collected, this is still the latest compilation which gives interpolated basement depths over the entire region, to our knowledge.

L608 “this means adding gravity data when there is very little”. To rephrase.
 > Will do.

L629 “skewness” Are you talking about the asymmetry of the distribution of the constraint points?
 > Yes, specifically the asymmetry of the distances of each grid cell to the nearest constraint point, essentially the asymmetry of the colorbar histograms of Figure 6d, for example.

L643 “won’t” to replace by “will not”.
 > Will do.

L721 “Gravity inversion” to replace by “Free-air gravity-based bathymetry inversions”?
 > Will do.

L722-723 “Using synthetic tests based on real bathymetry data from the Ross Sea” and using Antgg2021, right? To be added in the sentence. See comment on test location in specific comments.
 > No, AntGG was only used in the ice shelf analysis portion.

## Figure 1.
Above ellipsoid: “ρ earth-ρ air”: Maybe I didn't understand, but for the bedrock above the ellipsoid and below the ice, isn't the equation supposed to be something like “ρ earth + ρ ice – ρ air”?
Put a full black contour around your color legends.
 > Ok will add a black contour, thanks. If you are referring to subplot d, this should the "anomalous" density, relative the Normal Earth. So it's the density relative to air for any point above the ellipsoid and the density relative to rock for anything below the ellipsoid. If your referring to subplot h, the end product should be a density of "ρ earth – ρ air", since it is rock where air should be. But to achieve that, we have three overlapping prisms (light grey), which when combined give the correct density values (black). The three prism sets have the following densities "ρ earth – ρ water", "ρ water – ρ ice", and "ρ ice – ρ air", which you'll see if you add them will give "ρ earth – ρ air".

## Figure 2.
 What is “Normal gravity” in your input data?
 > Normal gravity is the mathematically calculated gravity effect of the WGS-84 ellipsoidal model of the Earth, use the analytical solutions of Nagy et al. 2000.

## Figure 3.
“One prism layer either above (c) or below (b)”. To replace “or below (b)” by “or below (d)”.
 > Will do.

## Figure 7.
Some ice shelf names are missing on a) and b) like Shackleton, West, Ronne.
 > Good catch, thanks, we will add the other labels.
“24 previously inverted shelves are in red”. Read Charrassin et al., 2024. All ice shelves were already inverted (even the ones in black).
 > We will update the caption to read "the 24 previously individually-inverted shelves ...", similar to the  caption in Figure 16.

## Figure 12.
“e)”. Describe before d), as d) is the difference between c) and e). We need to know what is e) before knowing what is d).
 > We put them in this order so e and f can be side-by-side and easily compared.
“true regional component of the misfit”. How can you compute that?
 > Since this is all synthetic data, we now the true regional signal since it was the forward-calculated gravity effect of the basement topography model.

## Figure 16.
Where does come from the RMSE for each ice shelf? Why do you have two values per ice shelf? Is it the min/max?
 > The RMSE is predicted from where the ice shelf falls on subplot a. To know where the shelf falls on the plot, we estimated the x and y axis values using the existing bed constraint points and AntGG. In the legend, the first value is the RMSE, from where the shelf is plotted on subplot a, and the second value is the topo-improvement, from where the shelf is plotted on subplot b. We can try and clarify this in the caption.

## Figure 18.
The captions of the plots are illegible.
 > We will increase the size.
