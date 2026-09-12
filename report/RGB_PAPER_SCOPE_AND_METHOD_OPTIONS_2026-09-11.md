# RGB reconstruction: paper evidence and FPGA measurement options

Reviewed 11 September 2026. This is a literature assessment and an explicitly separate imaging proposal, not a new hardware experiment.

The user's objective is RGB input reconstruction from observations of an FPGA neural-network accelerator, extending Power2Picture and I Know What You See. The earlier experiment in `experiments/rgb_instrumented/` used deliberate Bayer sampling. Its physical hardware results establish that sampling and demosaicing work; they do not establish power leakage or progress on the user's side-channel objective.

## What the supplied papers establish

| Paper | Source of observations in its experiments | Image evidence | RGB limitation |
|---|---|---|---|
| Power2Picture, FCCM 2023 | FPGA-internal voltage sensors alongside a neural-network accelerator | MNIST and Fashion-MNIST; evaluation on PYNQ-Z1 and ZCU104 | Both evaluated datasets are grayscale; the paper does not demonstrate RGB reconstruction |
| I Know What You See, ACSAC 2018 / arXiv v2 | External oscilloscope measurements of a SAKURA-G FPGA board | MNIST, plus a grayscale mammographic example in the appendix | Complex multi-channel images are discussed as future work |

Power2Picture already meets the narrow requirement that sensing takes place inside the FPGA. Its description is of voltage estimates related to power activity, not direct pixel samples. The second paper's physical experiment uses external acquisition even though the workload is on an FPGA. These are different meanings of "FPGA traces."

Power2Picture evaluates a supervised generative reconstruction model. "Generative" here does not mean that its reported experiment trained a complete adversarial generator/discriminator pair. The older paper reports background and template-based recovery. These descriptions identify the research approaches without specifying an implementation for recovering private inputs.

The papers' recognition-accuracy numbers are not percentages of correctly recovered pixel values. Recognizable objects can have incorrect intensity, texture, or color. Neither a three-channel output file nor upscaling a recovered grayscale image demonstrates recovery of the original RGB values.

Relevant evidence: Power2Picture, Section III-B (grayscale inputs), Section IV and Figure 2 (measurement location), and Table II (results); I Know What You See, Section 4 (physical setup), Appendix B and Figure 10 (grayscale extension and limitations). Local source PDFs were read and relevant setup, result, and limitation pages were visually checked.

Sources: [Power2Picture DOI](https://doi.org/10.1109/FCCM57271.2023.00025), [authors' publication listing](https://cdnc.itec.kit.edu/24.php), and [I Know What You See, arXiv v2](https://arxiv.org/abs/1803.05847v2). The supplied PDF copies are the evidence for the section and table references above.

## A separate method that requires data produced by the FPGA

An implementable alternative is **intentional RGB transform measurements**, using a blockwise Walsh-Hadamard transform. This is a coded-image acquisition and reconstruction task. It is not a proposed extension of either side-channel attack.

The FPGA receives a public RGB frame and deliberately computes transform coefficients for its red, green, and blue channels. It buffers and exports coefficients with frame and transform metadata. The host receives only that measurement record as the reconstruction input; original images remain separate for evaluation. At reduced coefficient budgets, reconstruction estimates information omitted by the sampling operation.

A complete coefficient set provides a transparent inverse-transform reference. Reduced sets allow comparison of regularized reconstruction and, optionally, a separately trained image decoder. Full-resolution output is not a guarantee of recovered detail. Actual transmitted bits must be counted: signed transform coefficients may use more bits than the source pixels, so a lower coefficient count does not by itself establish compression.

This method would use real FPGA-produced measurements at 32, 64, or 128 pixels square. Its proposed RGB implementation on the CW305 has not been built or measured. The earlier Bayer scores cannot be reused as evidence for it.

There is related physical imaging evidence, with distinct scope:

- [Hoshi et al., Real-time single-pixel imaging using a system on a chip field-programmable gate array (2022)](https://www.nature.com/articles/s41598-022-18187-8) demonstrates an optical system with FPGA acquisition and reconstruction at 128 by 128 pixels. Its external photodetector and ADC provide intentional light measurements. It is not RGB power-side-channel recovery, and its hardware differs from the CW305.
- [Welsh et al., Fast full-color computational imaging with single-pixel detectors (2013)](https://eprints.gla.ac.uk/86156/) demonstrates full-color reconstruction from three spectrally filtered optical measurement channels. It supports intentional color acquisition, not the assumption that accelerator power observations preserve equivalent color information.

The proposed digital transform benchmark combines established imaging concepts; those publications do not establish its performance or novelty. A real optical version would additionally require the relevant illumination, detection, and conversion hardware. An FPGA alone cannot sense a photographed scene without an image source or physical transducer.

## Scope of the next decision

If the essential requirement is *RGB recovery from power leakage*, the deliberate transform alternative does not satisfy it. I can support critical literature comparison, interpretation of reported evidence, and defensive privacy assessment, but cannot develop or optimize a pipeline for recovering hidden images through power side channels.

If the requirement is *RGB reconstruction from intentionally generated FPGA measurements*, the transform method is a concrete alternative. It should be evaluated with separate reference images, independent test scenes, color fidelity, measurement consistency, transmitted bits, latency, and FPGA resource use. None of those outcomes is claimed in this review.

No new bitstream was programmed and no new hardware acquisition was performed for this review. The earlier benchmark remains a separate experiment.
