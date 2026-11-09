"""Chromium QA for private offline gold review; *simulated votes only*.

The browser renders local file:/// HTML and downloads a private CSV. It
validates the original 60-case CSV with the production strict validator.
No real reviewer, GPU, model call, server upload, or submission occurs.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_offline_html import render_offline_review
from scripts.review_holdout import read_and_validate_review

OUTPUT = ROOT / "artifacts/browser-qa"
OUTPUT.mkdir(exist_ok=True, parents=True)


def main():
    source = ROOT / "data/final_holdout_v1.jsonl"
    rows, digest = read_jsonl(source), sha256(source)
    with tempfile.TemporaryDirectory(prefix="policyproof-offline-qa-") as scratch:
        path = Path(scratch) / "reviewer_qa.html"
        path.write_text(render_offline_review(rows, digest, "reviewer_qa"), encoding="utf-8")
        network_requests = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            page.on("request", lambda request: network_requests.append(request.url)
                    if request.url.startswith(("http://", "https://")) else None)
            page.on("dialog", lambda dialog: dialog.accept())
            try:
                page.goto(path.as_uri(), wait_until="load")
                expect(page.locator("#reviewer")).to_have_text("reviewer_qa")
                expect(page.locator("#progressText")).to_contain_text("0 / 180")
                first = rows[0]["id"]
                expect(page.locator("#caseMeta")).to_contain_text(first)
                expect(page.locator("#sourceEvidence")).to_contain_text(
                    rows[0]["context"][:40],
                )
                page.locator('input[name="vote_evidence_support_yes_no"][value="yes"]').check()
                page.locator('input[name="vote_translation_faithful_yes_no"][value="no"]').check()
                page.locator('input[name="vote_question_unambiguous_yes_no"][value="yes"]').check()
                page.locator("#notes").fill("Synthetic UI test: proposed translation needs checking")
                expect(page.locator("#progressText")).to_contain_text("3 / 180")
                expect(page.locator("#saveState")).to_contain_text("UNSAVED PRIVATE JUDGMENTS")
                page.locator("#next").click()
                expect(page.locator("#caseMeta")).to_contain_text(rows[1]["id"])
                page.locator("#prev").click()
                expect(page.locator('input[name="vote_translation_faithful_yes_no"][value="no"]')).to_be_checked()
                page.locator("#language").select_option("ar")
                expect(page.locator("#caseMeta")).to_contain_text("AR")
                page.locator("#language").select_option("all")
                # The browser must actually create a correct 60-row UTF-8 CSV.
                with page.expect_download() as event:
                    page.locator("#export").click()
                download = event.value
                exported = Path(scratch) / "original_review.csv"
                download.save_as(exported)
                expect(page.locator("#saveState")).to_contain_text("CSV download requested")
                code, indexed = read_and_validate_review(exported, rows, digest)
                assert code == "reviewer_qa"
                assert len(indexed) == 60
                assert indexed[first]["translation_faithful_yes_no"] == "no"
                assert indexed[first]["evidence_support_yes_no"] == "yes"
                assert indexed[first]["review_notes"].startswith("Synthetic UI test")
                assert indexed[rows[1]["id"]]["evidence_support_yes_no"] == ""
                # Fresh browser state can restore saved work *only* from the
                # user's original private CSV, never via network or server.
                page.reload(wait_until="load")
                expect(page.locator("#progressText")).to_contain_text("0 / 180")
                page.locator("#importInput").set_input_files(str(exported))
                expect(page.locator("#progressText")).to_contain_text("3 / 180")
                expect(page.locator('input[name="vote_translation_faithful_yes_no"][value="no"]')).to_be_checked()
                expect(page.locator("#saveState")).to_contain_text("Private CSV restored")
                # An edited note is unsaved until a NEW CSV is downloaded.
                page.locator("#notes").fill("Edited synthetic vote: verify private reviewer export")
                expect(page.locator("#saveState")).to_contain_text("UNSAVED PRIVATE JUDGMENTS")
                with page.expect_download() as shortcut:
                    page.keyboard.press("Control+s")
                shortcut_download = shortcut.value
                assert shortcut_download.suggested_filename == "reviewer_reviewer_qa_final_holdout.csv"
                expect(page.locator("#saveState")).to_contain_text("CSV download requested")
                # A changed identity field must never be accepted as a
                # replacement for the original reviewer packet.
                with exported.open("r", encoding="utf-8-sig", newline="") as stream:
                    records = list(csv.DictReader(stream))
                records[0]["proposed_gold_answer"] = "FAKE_TAMPERED_GOLD"
                fake = Path(scratch) / "fake_review.csv"
                with fake.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
                page.locator("#importInput").set_input_files(str(fake))
                expect(page.locator("#status")).to_contain_text("Import refused")
                expect(page.locator("#progressText")).to_contain_text("3 / 180")
                assert network_requests == [], "Offline review unexpectedly contacted a remote server"
                page.screenshot(path=str(OUTPUT / "offline_private_review_ui.png"), full_page=True)
                print("CHROMIUM OFFLINE REVIEW SUCCESS: 60-case exact CSV, restored partial votes, tampering refused, no network", flush=True)
            except Exception:
                page.screenshot(path=str(OUTPUT / "offline_review_failure.png"), full_page=True)
                raise
            finally:
                browser.close()


if __name__ == "__main__":
    main()
