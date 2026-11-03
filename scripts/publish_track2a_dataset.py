"""Optional Hugging Face dataset publication. OFF by default.

Validates original 20-file research ZIP, all original SHA256 hashes and honest
48-evaluated/72-unscored labels. No GPU/model calls. Token ONLY in HF_TOKEN
environment variable on owner's machine. No token is printed or committed.
Does NOT invent a fine-tuned model, adapter, or public model collection.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from zipfile import ZipFile

EXPECTED_OFFICIAL_ARCHIVE_SHA256="d00832984b6d119fca8f048e470427bf522a14432ef92c99de6c7e35ccf0bdc7"
LICENSES={"cc-by-4.0","cc0-1.0","cc-by-sa-4.0"}
REQUIRED={
    "README.md","provenance/SHA256SUMS.txt","provenance/PACKAGE_STATUS.json",
    "challenge_72/challenge_v1_UNSCORED.jsonl",
    "challenge_72/challenge_manifest_UNSCORED.json",
    "evaluated_48/original_saved_tiny_aya_base_model_predictions.json",
    "evaluated_48/curator_audit_chain_report.json",
    "evaluated_48/auditkit_native_receipt.json",
}
NONPUBLIC={"LICENSE_REVIEW_REQUIRED.md","PUBLISHING_INSTRUCTIONS.md",
           "provenance/PACKAGE_STATUS.json","provenance/SHA256SUMS.txt"}
SECRETS=[re.compile(rb"hf_[a-zA-Z0-9]{30,}"),
         re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
         re.compile(rb"(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{35,}")]
HF_REPO=re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")


def validate_archive(archive:Path, *, expected_sha256:str|None=EXPECTED_OFFICIAL_ARCHIVE_SHA256)->dict[str,bytes]:
    # The shipping path pins the exact 20-file owner-reviewed archive. Only
    # synthetic unit tests override this hash with their fixture bytes.
    if expected_sha256 is not None and sha256(archive.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError("Research ZIP hash differs from the verified public handoff")
    files={}
    with ZipFile(archive) as z:
        if len(z.infolist())>100 or z.testzip() is not None:
            raise ValueError("Corrupt or unexpectedly large ZIP")
        total=0
        for info in z.infolist():
            if info.is_dir():continue
            name=info.filename
            parts=PurePosixPath(name).parts
            if name.startswith("/") or "\\" in name or ".." in parts or name in files:
                raise ValueError("Unsafe or duplicate ZIP entry")
            if (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError("Symlink ZIP entries forbidden")
            if info.file_size>5_000_000:raise ValueError("Unexpected large entry")
            total+=info.file_size
            if total>25_000_000:raise ValueError("ZIP expansion limit reached")
            value=z.read(info)
            if any(pattern.search(value) for pattern in SECRETS):
                raise ValueError("Potential secret found in "+name)
            files[name]=value
    if not REQUIRED.issubset(files) or (expected_sha256 is not None and len(files)!=20):
        raise ValueError("Expected exactly the verified 20-file research handoff")
    expected={}
    for line in files["provenance/SHA256SUMS.txt"].decode("utf-8").splitlines():
        m=re.fullmatch(r"([a-f0-9]{64})  ([^\n]+)",line)
        if not m or m.group(2) in expected:
            raise ValueError("Invalid or duplicate hash manifest row")
        expected[m.group(2)]=m.group(1)
    if set(expected)!=set(files)-{"provenance/SHA256SUMS.txt","provenance/PACKAGE_STATUS.json"}:
        raise ValueError("Incomplete original checksum coverage")
    for name,value in expected.items():
        if sha256(files[name]).hexdigest()!=value:
            raise ValueError("Original checksum mismatch: "+name)
    status=json.loads(files["provenance/PACKAGE_STATUS.json"])
    if (status.get("status")!="local_hf_upload_staging_ONLY_not_published"
        or status.get("evaluated_cases")!=48
        or status.get("new_unscored_cases")!=72
        or status.get("selected_license") is not None):
        raise ValueError("Unexpected status or sample counts")
    baseline=json.loads(files[
        "evaluated_48/original_saved_tiny_aya_base_model_predictions.json"])
    if len(baseline.get("predictions",[]))!=48 or baseline.get("is_finetuned") is not False:
        raise ValueError("Original model evidence changed")
    new=[json.loads(x) for x in files[
        "challenge_72/challenge_v1_UNSCORED.jsonl"].decode("utf-8").splitlines() if x.strip()]
    if len(new)!=72 or len({x["id"] for x in new})!=72:
        raise ValueError("Original 72 challenge cases changed")
    if any(r.get("model_evaluated") is not False
           or r.get("human_reviewed") is not False
           or r.get("used_for_training") is not False for r in new):
        raise ValueError("Unscored/independent-review status changed")
    return files


def stage(files:dict[str,bytes],folder:Path,license_id:str)->list[str]:
    if license_id not in LICENSES:raise ValueError("Invalid/undecided license")
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise FileExistsError("Refusing to overwrite staging files")
    result={name:value for name,value in files.items() if name not in NONPUBLIC}
    card=files["README.md"].decode("utf-8")
    old="**Research draft, NOT YET PUBLISHED on Hugging Face.** This ZIP is staged for the dataset owner to review, select a license, and upload after authenticating to their own HF account."
    if old not in card or not card.startswith("---\n") or "## Licensing\n" not in card:
        raise ValueError("Unexpected original HF dataset card")
    card=card.replace("---\n","---\nlicense: "+license_id+"\n",1).replace(
        old,"**Public synthetic research dataset.** The publisher confirmed dataset rights and publication scope."
    )
    card=card.split("## Licensing\n")[0]+"## Licensing\nThe publisher affirmed rights and selected the "+license_id+" license. Original gated Tiny Aya weights are not redistributed.\n"
    result["README.md"]=card.encode("utf-8")
    result["provenance/PUBLICATION_SCOPE.json"]=(json.dumps({
        "selected_license":license_id,
        "evaluated_base_model_cases":48,
        "new_unscored_model_cases":72,
        "independent_human_review_completed":False,
        "tuned_adapter_included":False,
        "original_archive_checksums_sha256":sha256(files["provenance/SHA256SUMS.txt"]).hexdigest()
    },indent=2)+"\n").encode("utf-8")
    result["provenance/SHA256SUMS.txt"]=("\n".join(
        sha256(value).hexdigest()+"  "+name
        for name,value in sorted(result.items()))+"\n").encode("utf-8")
    for name,value in result.items():
        path=folder/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value)
    return sorted(result)


def publish(folder:Path,filenames:list[str],repo_id:str,token:str,
            *,new_collection=False,collection_slug=None,allow_existing=False,
            api=None,anonymous_api=None)->dict:
    if api is None or anonymous_api is None:
        from huggingface_hub import HfApi
        api=api or HfApi(token=token)
        anonymous_api=anonymous_api or HfApi(token=False)
    if not allow_existing:
        api.create_repo(repo_id=repo_id,repo_type="dataset",private=False,exist_ok=False)
    if getattr(api.repo_info(repo_id=repo_id,repo_type="dataset"),"private",True):
        raise RuntimeError("Dataset repository is private; refusing upload")
    commit=api.upload_folder(repo_id=repo_id,repo_type="dataset",folder_path=str(folder),
                             commit_message="PolicyProof Track 2a synthetic safety benchmark")
    info=anonymous_api.repo_info(repo_id=repo_id,repo_type="dataset",token=False)
    if getattr(info,"private",True):
        raise RuntimeError("Repository not publicly readable by anonymous client")
    observed=set(anonymous_api.list_repo_files(repo_id=repo_id,repo_type="dataset",token=False))
    if not set(filenames).issubset(observed):
        raise RuntimeError("Expected dataset files are absent from public repo")
    anonymous_api.hf_hub_download(repo_id=repo_id,repo_type="dataset",
                                  filename="README.md",token=False)
    if new_collection:
        collection=api.create_collection(
            title="PolicyProof AI — Multilingual Safety Datasets",
            namespace=repo_id.split("/")[0],private=False,
            description="Synthetic 48 evaluated + 72 unscored Track 2a research")
        collection_slug=collection.slug
    api.add_collection_item(collection_slug=collection_slug,item_id=repo_id,
                            item_type="dataset",exists_ok=True)
    public=anonymous_api.get_collection(collection_slug,token=False)
    if getattr(public,"private",True) or not any(
        i.item_id==repo_id and i.item_type=="dataset" for i in public.items
    ):
        raise RuntimeError("Collection membership not anonymously verifiable")
    return {
        "public_dataset_repo":"https://huggingface.co/datasets/"+repo_id,
        "public_dataset_collection":"https://huggingface.co/collections/"+collection_slug,
        "anonymous_repo_and_collection_verified":True,
        "uploaded_file_count":len(filenames),
        "commit_sha":str(getattr(commit,"oid","unknown")),
        "model_collection_created":False,
        "official_lexsi_submission_completed":False,
    }


def validate_flags(args)->None:
    if args.repo_id is not None and not HF_REPO.fullmatch(args.repo_id):
        raise ValueError("Repo must be HF_NAMESPACE/DATASET_SLUG")
    if args.publish:
        if (not args.confirm_rights or not args.confirm_publication
            or not args.repo_id or bool(args.create_collection)==bool(args.collection_slug)):
            raise ValueError("Publication requires explicit rights, public consent, repo ID, and one collection mode")
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("HF_TOKEN is absent from the private user environment")
    elif args.create_collection or args.collection_slug:
        raise ValueError("Collection options require --publish")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--license",required=True,choices=sorted(LICENSES))
    p.add_argument("--repo-id")
    p.add_argument("--publish",action="store_true")
    p.add_argument("--confirm-rights",action="store_true")
    p.add_argument("--confirm-publication",action="store_true")
    p.add_argument("--create-collection",action="store_true")
    p.add_argument("--collection-slug")
    p.add_argument("--allow-existing-repo",action="store_true")
    p.add_argument("--receipt",type=Path,default=Path("track2a_hf_receipt.json"))
    args=p.parse_args()
    validate_flags(args)
    files=validate_archive(args.archive)
    with tempfile.TemporaryDirectory(prefix="policyproof_review_") as tmp:
        filenames=stage(files,Path(tmp),args.license)
        print(json.dumps({"status":"dry_run_verified" if not args.publish else "publishing",
                          "source_files":len(files),"staged_public_files":len(filenames),
                          "48_evaluated":True,"72_evaluated":False,
                          "license":args.license,"network_used":bool(args.publish)},indent=2))
        if not args.publish:return
        if args.receipt.exists():
            raise FileExistsError("Local receipt already exists; refusing overwrite")
        result=publish(Path(tmp),filenames,args.repo_id,os.environ["HF_TOKEN"],
                       new_collection=args.create_collection,
                       collection_slug=args.collection_slug,
                       allow_existing=args.allow_existing_repo)
        args.receipt.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
