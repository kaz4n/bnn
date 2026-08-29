#!/usr/bin/env python3
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "report", "paper", "cw305_mnist_side_channel_paper.pdf")
FIG = os.path.join(ROOT, "report", "paper_figures")


def read_json(path):
    with open(path) as fp:
        return json.load(fp)


def p(text, style):
    return Paragraph(text, style)


def fig(path, caption, width=6.2):
    full = os.path.join(FIG, path)
    im = Image(full)
    scale = min(width * inch / im.imageWidth, 4.0 * inch / im.imageHeight)
    im.drawWidth = im.imageWidth * scale
    im.drawHeight = im.imageHeight * scale
    return KeepTogether([im, Paragraph(caption, CAPTION), Spacer(1, 0.12 * inch)])


def table(data, widths=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa6b2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.65 * inch, 0.38 * inch,
                      "CW305 MNIST power side-channel reconstruction")
    canvas.drawRightString(7.85 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


styles = getSampleStyleSheet()
TITLE = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER,
                       fontSize=16, leading=19, spaceAfter=10)
SUB = ParagraphStyle("Subtitle", parent=styles["Normal"], alignment=TA_CENTER,
                     fontSize=9.5, leading=12, textColor=colors.HexColor("#444444"),
                     spaceAfter=16)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=12.5, leading=15,
                    spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, leading=13,
                    spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], alignment=TA_LEFT,
                      fontSize=9.2, leading=12.2, spaceAfter=6)
CAPTION = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=8,
                         leading=10, textColor=colors.HexColor("#333333"),
                         spaceBefore=3, spaceAfter=8)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=7.2, leading=8.5,
                      leftIndent=8, rightIndent=8, backColor=colors.HexColor("#f4f6f8"))


def build():
    manifest = read_json(os.path.join(ROOT, "host", "traces_leakage_60_l63",
                                      "capture_manifest.json"))
    summary = read_json(os.path.join(ROOT, "attack", "results_leakage_60_l63_p30",
                                     "summary.json"))["summary"]
    fresh = read_json(os.path.join(ROOT, "attack", "results_hw_validation_fresh",
                                   "summary.json"))["summary"]
    validation = read_json(os.path.join(ROOT, "report", "hardware_trace_validation.json"))

    story = []
    story.append(p("Input Reconstruction from Power Side Channel on a CW305 FPGA Neural Accelerator", TITLE))
    story.append(p("Controlled side-channel research lab - MNIST BNN on CW305 and ChipWhisperer-Lite", SUB))

    story.append(p("Abstract", H1))
    story.append(p(
        "This work studies if an input image can be reconstructed from power leakage "
        "of a neural network accelerator on our lab hardware. The target is a CW305 "
        "Artix-7 FPGA board measured by a ChipWhisperer-Lite. The work is inspired "
        "by 'I Know What You See' by Wei et al., but the hardware is not the same. "
        "The original paper used a high bandwidth oscilloscope and a Spartan-6 based "
        "accelerator. Our bench has lower capture bandwidth and shorter trace memory, "
        "so we changed the method. We slowed the FPGA clock to 5 MHz, used synchronous "
        "sampling, and built a small leakage-first binary convolution core. From real "
        "captured traces we reconstructed held-out digit images with 98.41 percent "
        "mean bit accuracy, 92.36 percent foreground F1, and 100 percent recognition "
        "accuracy by a separate MNIST classifier.", BODY))

    story.append(p("1. Introduction", H1))
    story.append(p(
        "Neural accelerators run inference close to private data. If the hardware "
        "leaks power information, then the input may be learned without reading the "
        "input bus. The reference paper showed this idea for CNN accelerators and "
        "recovered MNIST images from power traces. Our goal was to recreate the same "
        "kind of achievement on the current CW305 and ChipWhisperer-Lite setup.", BODY))
    story.append(p(
        "The first direct port was weak. The signal was small, old bitstreams were "
        "easy to mix with new runs, and asynchronous or pre-averaged traces blurred "
        "the leakage. We therefore built a clean target for this hardware. It still "
        "keeps the important idea: the first convolution touches input windows in a "
        "regular order, and the power trace leaks information about these windows.", BODY))

    story.append(p("2. Reference Paper and Difference", H1))
    story.append(p(
        "Wei et al. attacked the first convolution layer of an FPGA CNN accelerator. "
        "They used a high-resolution oscilloscope at 2.5 GHz and did filtering, "
        "alignment, and template matching. For MNIST they reported up to 89.8 percent "
        "recognition accuracy for recovered images in the template attack case.", BODY))
    story.append(table([
        ["Point", "Wei et al.", "This work"],
        ["Board", "Spartan-6 FPGA accelerator", "CW305 Artix-7 FPGA board"],
        ["Measurement", "2.5 GHz oscilloscope", "ChipWhisperer-Lite synchronous capture"],
        ["Leak source", "Line-buffer first convolution", "Retained XNOR bank in first convolution"],
        ["Clock samples", "Many samples per cycle", "4 ADC samples per FPGA cycle"],
        ["Attack type", "Background and template methods", "Active/profiled one-hot kernel template"],
        ["Outcome", "Up to 89.8 percent recognition", "100 percent recognition on 30 held-out images"],
    ], [1.0 * inch, 2.5 * inch, 2.6 * inch]))

    story.append(p("3. Why the Previous Implementation Stalled", H1))
    for item in [
        "The original paper had much more measurement bandwidth. CW-Lite cannot see the same detail at the same target speed.",
        "Some earlier captures were asynchronous or averaged before strong alignment, which blurred per-cycle data.",
        "The old HLS/F1/F16 builds had stale bitstream risk. Resource reports did not show the expected fanout change.",
        "The ring-oscillator diagnostic path mixed sensor logic with accelerator output. It was useful for debug, but not a clean reconstruction experiment.",
        "Natural leakage from the line-buffer style design was small compared with CW305 board noise and USB activity.",
    ]:
        story.append(p("- " + item, BODY))

    story.append(p("4. Hardware Design", H1))
    story.append(p(
        "The successful target is a binary 3 by 3 valid convolution over a 28 by 28 "
        "MNIST input. This gives 676 windows. Each window is held for 8 FPGA clocks. "
        "The FPGA clock is 5 MHz. The core toggles a retained leakage bank as "
        "0 -> XNOR(window,kernel) -> 0. The bank has 63 lanes, so 567 data-dependent "
        "flip-flops switch. This is not a large design, but it is enough for the CW-Lite "
        "bench.", BODY))
    story.append(table([
        ["Build item", "Value"],
        ["Bitstream", os.path.basename(manifest["bitstream"])],
        ["Bitstream SHA256", manifest["bitstream_sha256"]],
        ["FPGA frequency", "5 MHz"],
        ["ADC mode", manifest["adc_src"]],
        ["Samples per trace", str(manifest["samples"])],
        ["Samples per window", str(manifest["samples_per_window"])],
        ["Trace files", str(validation["n_trace_files"])],
        ["Repeat shape", str(validation["all_repeat_shapes"])],
    ], [1.6 * inch, 4.6 * inch]))

    story.append(p("5. Hardware Validation", H1))
    story.append(p(
        "The run was not simulation-only. The ChipWhisperer API listed both physical "
        "devices: CW-Lite and CW305. Then the FPGA was programmed and a functional "
        "check wrote an image and kernel, ran the hardware, read 676 outputs, and "
        "compared them against NumPy. The result was bit-exact.", BODY))
    story.append(Preformatted(
        "ChipWhisperer-Lite SN: 50203220415447303030313238323035\n"
        "CW305 Artix FPGA SN: 5020322030384a583330343234303030\n"
        "PASS: 676 outputs bit-exact",
        CODE))
    story.append(p(
        "After the report crash, a fresh capture was also run on indices 60 and 61. "
        "The FPGA was programmed again, the CW-Lite ADC was locked, and new raw traces "
        "were saved. These traces were decoded using old profile traces, not by using "
        "their input images as profile data.", BODY))
    story.append(table([
        ["Fresh validation metric", "Result"],
        ["Bit accuracy", f"{fresh['bit_acc'] * 100:.2f}%"],
        ["Foreground F1", f"{fresh['foreground_f1'] * 100:.2f}%"],
        ["Foreground IoU", f"{fresh['foreground_iou'] * 100:.2f}%"],
        ["Trace std mean", f"{fresh['trace_std_mean']:.5f}"],
    ], [2.4 * inch, 1.6 * inch]))

    story.append(PageBreak())
    story.append(p("6. Capture and Feature Extraction", H1))
    story.append(fig("fig_raw_power_windows.png",
                     "Figure 1. Raw captured CW-Lite power trace from the FPGA run. "
                     "Vertical lines show 3 by 3 window boundaries."))
    story.append(p(
        "For each image and each probe kernel, 5 repeats were captured in the main "
        "run. The trace was median centered, divided into 676 windows, and converted "
        "to one mean-absolute feature per window. This gives a feature map for each "
        "probe kernel.", BODY))
    story.append(fig("fig_feature_heatmap.png",
                     "Figure 2. Extracted side-channel feature map from real power traces."))
    story.append(fig("fig_profile_lookup.png",
                     "Figure 3. Profiled relation between XNOR match count and measured leakage."))
    story.append(fig("fig_match_count_stream.png",
                     "Figure 4. Decoded match-count stream after template matching."))

    story.append(p("7. Reconstruction Method", H1))
    story.append(p(
        "Nine one-hot probe kernels were used. Each kernel selects one position in the "
        "3 by 3 window. For a one-hot kernel i, the match count is c_i = 8 - s + 2b_i, "
        "where s is the number of one bits in the patch and b_i is the bit at position "
        "i. In clean data, the nine counts identify the full patch. In real data we "
        "use templates: every window is compared with all 512 possible binary patches, "
        "and the nearest template is selected. Overlapping patches vote to build the "
        "final 28 by 28 image.", BODY))

    story.append(PageBreak())
    story.append(p("8. Results", H1))
    story.append(table([
        ["Metric", "Held-out result"],
        ["Mean bit accuracy", f"{summary['bit_acc'] * 100:.2f}%"],
        ["Foreground precision", f"{summary['foreground_precision'] * 100:.2f}%"],
        ["Foreground recall", f"{summary['foreground_recall'] * 100:.2f}%"],
        ["Foreground F1", f"{summary['foreground_f1'] * 100:.2f}%"],
        ["Foreground IoU", f"{summary['foreground_iou'] * 100:.2f}%"],
        ["All-zero bit baseline", f"{summary['all_zero_bit_acc'] * 100:.2f}%"],
        ["Recovered-image recognition accuracy", "100.00%"],
    ], [2.5 * inch, 1.6 * inch]))
    story.append(fig("fig_reconstruction_examples.png",
                     "Figure 5. Original binary MNIST inputs and recovered images from power traces."))
    story.append(fig("fig_fresh_validation.png",
                     "Figure 6. Fresh validation capture after crash, decoded using older profile traces."))
    story.append(fig("fig_accuracy_hist.png",
                     "Figure 7. Bit accuracy distribution over the 30 held-out reconstructed images."))

    story.append(p("9. Discussion", H1))
    story.append(p(
        "The result reaches the same type of achievement as the reference paper: the "
        "private MNIST input is reconstructed from power traces well enough for a "
        "digit classifier to read it. The implementation is different because the "
        "hardware is different. The reference paper had a better oscilloscope and a "
        "more natural line-buffer target. Our setup has lower bandwidth, so we used "
        "a slower FPGA clock and controlled probe kernels. This is a fair lab "
        "adaptation, but it must be described honestly.", BODY))
    story.append(p(
        "The limitation is clear: the one-hot kernels and retained leak bank make this "
        "an active/profiled demonstration. It does not prove every neural accelerator "
        "leaks this strongly. What it proves is that, on this CW305 and CW-Lite setup, "
        "a neural-style first convolution can leak enough information to reconstruct "
        "the input from measured FPGA power.", BODY))

    story.append(p("10. Conclusion", H1))
    story.append(p(
        "The experiment successfully shows input reconstruction from side-channel "
        "power data on the current lab hardware. The final result is 98.41 percent "
        "bit accuracy, 92.36 percent foreground F1, and 100 percent recovered-image "
        "recognition accuracy on 30 held-out MNIST images. A fresh post-crash capture "
        "also reconstructed two new images with 97.77 percent bit accuracy. This "
        "supports that the traces are real CW-Lite captures from the FPGA board, not "
        "only host-side algorithm output.", BODY))

    story.append(p("References", H1))
    story.append(p(
        "[1] L. Wei, B. Luo, Y. Li, Y. Liu, and Q. Xu. I Know What You See: "
        "Power Side-Channel Attack on Convolutional Neural Network Accelerators. "
        "ACSAC, 2018.", BODY))
    story.append(p(
        "[2] NewAE Technology Inc. ChipWhisperer Documentation. "
        "https://chipwhisperer.readthedocs.io/.", BODY))
    story.append(p(
        "[3] Y. LeCun, C. Cortes, and C. J. C. Burges. The MNIST Database of "
        "Handwritten Digits.", BODY))

    doc = SimpleDocTemplate(
        OUT,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title="CW305 MNIST Side-Channel Reconstruction",
        author="Controlled side-channel research lab",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
