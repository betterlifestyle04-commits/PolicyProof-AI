"""Owner-only offline HTML handoff for the official Lexsi submission form.

Renders only fields produced by the SOURCE-CHECKED local form-handoff module.
This file is neither a submitted project nor an automated browser/form client.
It never asks for or stores the owner's private registration email or HF token.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re

from policyproof.core import ROOT
from scripts.lexsi_submission_handoff import OFFICIAL_FORM, handoff

OUTPUT_REL = Path("artifacts/lexsi_track2a_owner_form_offline_UNSUBMITTED.html")
VALID_STATUS = {
    "DRAFT_ONLY_MISSING_PUBLICATION_OR_RESEARCH",
    "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED",
    "LOCAL_EVIDENCE_READY_MANUAL_FORM_REQUIRED",
}
SHA = re.compile(r"^[0-9a-f]{64}$")


def _safe(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected exact text field in Lexsi owner handoff")
    return escape(value, quote=True)


def _verify(packet: dict) -> dict:
    """Refuse false submission claims and unverified public collection links."""
    if not isinstance(packet, dict) or (
        packet.get("schema") != "policyproof_lexsi_track2a_owner_form_handoff_v1"
        or packet.get("status") not in VALID_STATUS
        or packet.get("official_form_url") != OFFICIAL_FORM
    ):
        raise ValueError("Unrecognized or untrusted local Lexsi form handoff")
    for key in (
        "official_form_opened_by_script", "email_lookup_performed",
        "registered_email_verified", "owner_license_and_rights_approved_by_script",
        "official_submission_completed", "organizer_confirmation_verified",
        "external_network_used",
    ):
        if packet.get(key) is not False:
            raise ValueError("No submitted, remote-approved or private-email claims allowed")
    if packet.get("registered_email") is not None:
        raise ValueError("Private registration email must never enter offline handoff")
    fields = packet.get("form_fields")
    if not isinstance(fields, dict):
        raise ValueError("Missing source-checked Lexsi form fields")
    approved_status = packet["status"] == "LOCAL_EVIDENCE_READY_MANUAL_FORM_REQUIRED"
    if packet.get("eligible_for_manual_form_review_locally") is not approved_status:
        raise ValueError("Contradictory local form readiness flags")
    body = fields.get("abstract_text")
    if body is not None:
        if not isinstance(body, str) or (
            not 50 <= len(body.split()) < 500
            or fields.get("abstract_word_count") != len(body.split())
            or fields.get("abstract_under_500_words") is not True
            or not SHA.fullmatch(str(fields.get("abstract_source_sha256", "")))
        ):
            raise ValueError("Unverified or oversized owner abstract")
    elif packet["status"] != "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED":
        raise ValueError("Unexpected missing non-stale historical abstract")
    elif fields.get("abstract_under_500_words") is not True:
        raise ValueError("Unverified source abstract history")
    if packet["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED" and body is not None:
        raise ValueError("Stale abstract must be withheld")

    hf_fields = ("hf_model_collection", "hf_dataset_collection", "hf_dataset_repository")
    public = fields.get("hf_anonymous_access_receipt_current")
    if public not in (True, False):
        raise ValueError("Unverified HF readback state")
    if public is False and any(fields.get(key) is not None for key in hf_fields):
        raise ValueError("Owner-entered/unverified HF links must never appear")
    if public is True and any(
        not isinstance(fields.get(key), str)
        or not fields[key].startswith("https://huggingface.co/")
        for key in hf_fields
    ):
        raise ValueError("HF links require source-checked canonical URLs")
    if not isinstance(packet.get("required_actions"), list) or not all(
        isinstance(item, str) and 0 < len(item) <= 500
        for item in packet["required_actions"]
    ):
        raise ValueError("Missing bounded owner action checklist")
    return fields


def _field(label: str, value: str | None, note: str) -> str:
    if value is None:
        return (
            '<div class="field"><div class="field-title">' + _safe(label)
            + '</div><p class="missing">MISSING — DO NOT INVENT A LINK</p>'
            + '<p class="hint">' + _safe(note) + '</p></div>'
        )
    return (
        '<div class="field"><label class="field-title">'
        + _safe(label)
        + '<input readonly aria-label="' + _safe(label)
        + '" value="' + _safe(value)
        + '" onclick="this.select()" /></label>'
        + '<p class="hint">' + _safe(note) + '</p></div>'
    )


def render_owner_form_html(packet: dict) -> str:
    """Pure, deterministic source-checked HTML. No scripts or hidden contacts."""
    fields = _verify(packet)
    blocked = packet["status"] != "LOCAL_EVIDENCE_READY_MANUAL_FORM_REQUIRED"
    abstract = fields["abstract_text"]
    if abstract is None:
        abstract_html = (
            '<p class="missing">BLOCKED: historical abstract is stale. '
            'Rewrite and source-check it before pasting anything into the official form.</p>'
        )
    else:
        abstract_html = (
            '<label class="field-title" for="abstract">Historical Track 2a abstract '
            '(' + str(fields["abstract_word_count"]) + ' of 499 maximum words)'
            '</label><textarea id="abstract" readonly spellcheck="false">'
            + _safe(abstract) + '</textarea>'
            '<p class="hint">Source SHA-256: '
            + _safe(fields["abstract_source_sha256"])
            + '. Review all claims again before pasting. This is a historical draft, '
            'not a certified model improvement.</p>'
        )
    details = "".join((
        _field("Project title", fields["project_title"],
               "Owner must confirm the final title matches the entry."),
        _field("Track", fields["track"],
               "Research track; no fine-tuning achievement is claimed."),
        _field("Problem statement", fields["problem_statement"],
               "Use only a problem statement supported by the study."),
        _field("Base model", fields["model_used_historically"],
               "Previously used base model; this is not a released adapted checkpoint."),
        _field("GitHub repository", fields["github_repository"],
               "URL was not independently anonymously accessed by this script."),
        _field("Write-up", fields["writeup_or_article"],
               "README is permitted; externally verify this URL before submission."),
        _field("Public demo", fields["public_demo"],
               "Saved-evidence observatory, not live fine-tuned inference; verify independently."),
        _field("Hugging Face model collection", fields["hf_model_collection"],
               "Requires public ungated model collection and human relevance/rights review."),
        _field("Hugging Face dataset collection", fields["hf_dataset_collection"],
               "Requires public dataset collection and source-bound anonymous readback."),
        _field("Hugging Face dataset repository", fields["hf_dataset_repository"],
               "Supplemental dataset link, only displayed after anonymous receipt verification."),
    ))
    toolkit = fields.get("toolkits_used_in_verified_chain")
    toolkit_text = ", ".join(toolkit) if isinstance(toolkit, list) and all(
        isinstance(x, str) for x in toolkit
    ) else "No source-checked toolchain available"
    checklist = "".join(
        "<li>" + _safe(item) + "</li>"
        for item in packet["required_actions"]
    )
    state = (
        "LOCAL EVIDENCE READY FOR OWNER REVIEW — NOT SUBMITTED"
        if not blocked else
        "BLOCKED / DRAFT ONLY — DO NOT SUBMIT AS COMPLETE"
    )
    return StringHTML(
        state=state,
        status=_safe(packet["status"]),
        abstract=abstract_html,
        details=details,
        toolkit=_safe(toolkit_text),
        checklist=checklist,
        deadline=_safe(packet["official_submission_deadline"]),
        official_url=_safe(OFFICIAL_FORM),
    )


def StringHTML(*, state: str, status: str, abstract: str, details: str,
               toolkit: str, checklist: str, deadline: str, official_url: str) -> str:
    """Static HTML and inline CSS only: no JavaScript, embedded credentials or forms."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
        'style-src &#39;unsafe-inline&#39;; script-src &#39;none&#39;; '
        'connect-src &#39;none&#39;; form-action &#39;none&#39;; base-uri &#39;none&#39;">'
        '<title>PolicyProof AI | Owner Lexsi form handoff — UNSUBMITTED</title>'
        '<style>'
        ':root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1d3046;background:#f3f6fa}'
        '*{box-sizing:border-box}body{margin:0;line-height:1.5}'
        'header{background:#152b43;color:#fff;padding:28px max(18px,calc((100vw - 1000px)/2))}'
        'h1{margin:0 0 6px;font-size:1.7rem}h2{font-size:1.2rem;margin-top:0}'
        'main{max-width:1000px;margin:24px auto;padding:0 18px 45px}'
        'section{background:white;border:1px solid #dce5ed;border-radius:12px;padding:22px;margin:16px 0}'
        '.state{border-left:5px solid #a45218;background:#fff5e8;padding:12px 16px;font-weight:700}'
        '.field{margin:18px 0}.field-title{display:block;font-weight:700;margin:0 0 5px}'
        'input,textarea{font:inherit;background:#f9fbfd;border:1px solid #a6b9c9;border-radius:6px;'
        'padding:10px;width:100%;color:#17324d;direction:auto}'
        'textarea{min-height:290px;white-space:pre-wrap;resize:vertical}'
        'input:focus,textarea:focus{outline:3px solid #89badc}'
        '.hint{color:#536679;font-size:.9rem;margin:5px 0}.missing{color:#9e350e;font-weight:700}'
        'li{margin:9px 0}a{color:#205b93}footer{font-size:.88rem;color:#526779}'
        '</style></head><body>'
        '<header><h1>PolicyProof AI · Lexsi owner form handoff</h1>'
        '<p>Offline source-checked draft · November 2026 · NOT an organizer receipt</p></header>'
        '<main><p class="state">' + state + ' · ' + status + '</p>'
        '<section><h2>Official deadline and next action</h2>'
        '<p>Deadline: <strong>' + deadline + '</strong>. '
        'After verifying rights, sources and links, open the '
        '<a target="_blank" rel="noopener noreferrer" href="' + official_url
        + '">official Lexsi submission form</a> yourself. '
        'This HTML does not log in, publish, submit, or certify eligibility.</p>'
        '<p><strong>Registration email:</strong> enter only in the actual organizer form. '
        'It is intentionally absent from this file.</p></section>'
        '<section><h2>Abstract (under 500 words)</h2>' + abstract + '</section>'
        '<section><h2>Source-bounded fields to check and copy manually</h2>'
        '<p>Read-only fields; select their text and copy it. Missing fields stay missing.</p>'
        + details + '<p><strong>Source-checked toolkits:</strong> ' + toolkit + '</p></section>'
        '<section><h2>Remaining actions that require the owner</h2><ol>'
        + checklist + '</ol></section>'
        '<section><h2>Research and submission boundaries</h2><p>'
        'Original 48-case development results are synthetic and do not establish '
        'general factual correctness. A separate 72-case challenge has no '
        'accepted scored model outcomes unless source replay later verifies them. '
        'Independent semantic review, license rights, public artifact relevance, '
        'and actual organizer acceptance require separate evidence. '
        'A saved HTML file or successful CI is not proof of submission.</p></section>'
        '<footer>No token, registration email, private reviewer judgment, '
        'browser persistence or automatic network call is included.</footer>'
        '</main></body></html>\n'
    )


def write_offline_html(root: Path = ROOT) -> Path:
    """Create one new directly scoped artifact; never overwrite a prior draft."""
    root = Path(root).resolve()
    target = root / OUTPUT_REL
    parent = target.parent
    if (parent.is_symlink() or (parent.exists() and not parent.is_dir())
        or target.exists() or target.is_symlink()):
        raise ValueError("Refusing to overwrite or follow an unsafe owner handoff")
    rendered = render_owner_form_html(handoff(root))
    parent.mkdir(exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        handle.write(rendered)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    target = write_offline_html()
    print(json.dumps({
        "status": "OFFLINE_OWNER_FORM_DRAFT_NOT_SUBMITTED",
        "output": str(target),
        "official_submission_completed": False,
        "external_network_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
