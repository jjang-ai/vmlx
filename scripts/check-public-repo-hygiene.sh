#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

forbidden=$(
  git ls-files |
    awk '
      /^docs\// &&
        !/^docs\/ARCHITECTURE\.md$/ &&
        !/^docs\/index\.md$/ &&
        !/^docs\/mlxstudio-releases-readme\.md$/ &&
        !/^docs\/api\// &&
        !/^docs\/benchmarks\// &&
        !/^docs\/development\/dsv4-decode-acceptance\.md$/ &&
        !/^docs\/development\/dsv4-encoder-contract\.md$/ &&
        !/^docs\/development\/(architecture|build-test-deploy|contributing)\.md$/ &&
        !/^docs\/getting-started\// &&
        !/^docs\/guides\// &&
        !/^docs\/reference\// {
          print
          next
        }
      index($0, "/") == 0 &&
        /\.md$/ &&
        !/^(README|CHANGELOG|CONTRIBUTING|SECURITY|CODE_OF_CONDUCT)\.md$/ {
          print
          next
        }
      /^notes\// ||
      /^PLANS\// ||
      /^autoresearch\// ||
      /^tmp\// ||
      /^build\// ||
      /^assets\/tools-tab\.png$/ ||
      /^panel\/(CHANGELOG|ENGINE-UPDATES|PROJECT|SETUP)\.md$/ ||
      /^productionapp\/(INSTALL|TECHNICAL-NOTES)\.md$/ ||
      /^vmlx_engine\/docs\/CODEBOOK-DEVELOPMENT\.md$/ ||
      /(^|\/)node_modules\// ||
      /^panel\/docs\/plans\// ||
      /^tests\/e2e\/results\// ||
      /^tests\/e2e\/panel-driver\/node_modules\// ||
      /^tests\/benchmark\/outputs\// ||
      /^tests\/e2e\/(AUDIT-REPORT|MATRIX|UI-SUITE)\.md$/ ||
      /^nohup\.out$/ ||
      /^trace_err2?\.txt$/ ||
      /^(gsm8k_qwen3_0\.6b_results|vlm_benchmark_results)\.json$/ ||
      /^vmlx_engine\/models\/minimax_m3\/(BUILD-STATUS|MASTER-STATUS|CAMPAIGN-CHECKLIST|CAMPAIGN-PROGRESS-LOG|M3-EAGLE3-NATIVE-MTP-HANDOFF|M3-MOE-QUANT-FIX-HANDOFF)\.md$/ ||
      /^vmlx_engine\/models\/minimax_m3\/MODEL-MATRIX-AUTODETECT\.txt$/ ||
      /^\.agents\// ||
      /^\.agent\// ||
      /^\.claude\// ||
      /^\.codex\// ||
      /^\.sisyphus\// ||
      /^\.factory\// ||
      /(^|\/)(botes|evidence|private-evidence|vmlx-proof|screenshots?|screen-recordings?|cdp-captures?|raw-sse|runtime-logs?)(\/|$)/ ||
      /(^|\/)[^\/]+\.(sqlite|db)$/ {
        print
      }
    '
)

if [ -n "$forbidden" ]; then
  printf '%s\n' \
    'ERROR: public repository contains forbidden private/internal artifacts:' \
    "$forbidden" >&2
  exit 1
fi

# Historical checks must describe the candidate release plus refs that are
# actually public. `git log --all` also walks stale local branches, refs from
# unrelated remotes, and local-only tags, so it can both misreport private
# scratch history as published and inspect tag histories that were rewritten
# on the public remote. Fetch origin's advertised branch/tag refs into a
# process-private namespace, inspect those exact objects, then remove the
# temporary refs. The object fetch is required because rewritten public tags
# may not exist under the checkout's stale local tag names.
public_remote=origin
public_ref_namespace="refs/vmlx-public-hygiene/$$"

cleanup_public_refs() {
  git for-each-ref --format='%(refname)' "$public_ref_namespace" |
    while IFS= read -r ref; do
      git update-ref -d "$ref"
    done
}
trap cleanup_public_refs EXIT
trap 'exit 1' HUP INT TERM

if ! git fetch --quiet --force --no-tags --no-write-fetch-head \
  "$public_remote" \
  "+refs/heads/*:$public_ref_namespace/heads/*" \
  "+refs/tags/*:$public_ref_namespace/tags/*"; then
  printf '%s\n' \
    "ERROR: unable to fetch public refs from remote $public_remote." >&2
  exit 1
fi

public_ref_commits=$(
  git for-each-ref \
    --format='%(objecttype) %(objectname) %(*objectname)' \
    "$public_ref_namespace" |
    awk '
      $1 == "commit" {
        print $2
        next
      }
      $1 == "tag" && $3 ~ /^[0-9a-f]+$/ {
        print $3
      }
    '
)

if [ -z "$public_ref_commits" ]; then
  printf '%s\n' \
    "ERROR: public remote $public_remote advertises no branch or commit tag refs." >&2
  exit 1
fi

public_history_commits=$(
  {
    git rev-parse --verify 'HEAD^{commit}'
    printf '%s\n' "$public_ref_commits"
  } |
    awk '/^[0-9a-f]+$/ {print}' |
    sort -u
)

if [ -z "$public_history_commits" ]; then
  printf '%s\n' 'ERROR: no candidate or public history commits were resolved.' >&2
  exit 1
fi

for commit in $public_history_commits; do
  if ! git cat-file -e "$commit^{commit}" 2>/dev/null; then
    printf '%s\n' \
      "ERROR: advertised public ref object $commit does not resolve to an inspectable commit." >&2
    exit 1
  fi
done

public_git_log() {
  printf '%s\n' "$public_history_commits" |
    git log --stdin --no-renames "$@"
}

historical_forbidden=$(
  public_git_log --name-only --format= |
    awk '
      /^docs\// &&
        !/^docs\/ARCHITECTURE\.md$/ &&
        !/^docs\/index\.md$/ &&
        !/^docs\/mlxstudio-releases-readme\.md$/ &&
        !/^docs\/api\// &&
        !/^docs\/benchmarks\// &&
        !/^docs\/development\/dsv4-decode-acceptance\.md$/ &&
        !/^docs\/development\/dsv4-encoder-contract\.md$/ &&
        !/^docs\/development\/(architecture|build-test-deploy|contributing)\.md$/ &&
        !/^docs\/getting-started\// &&
        !/^docs\/guides\// &&
        !/^docs\/reference\// {
          print
          next
        }
      index($0, "/") == 0 &&
        /\.md$/ &&
        !/^(README|CHANGELOG|CONTRIBUTING|SECURITY|CODE_OF_CONDUCT)\.md$/ {
          print
          next
        }
      /^notes\// ||
      /^PLANS\// ||
      /^autoresearch\// ||
      /^tmp\// ||
      /^build\// ||
      /^assets\/tools-tab\.png$/ ||
      /^panel\/(CHANGELOG|ENGINE-UPDATES|PROJECT|SETUP)\.md$/ ||
      /^productionapp\/(INSTALL|TECHNICAL-NOTES)\.md$/ ||
      /^vmlx_engine\/docs\/CODEBOOK-DEVELOPMENT\.md$/ ||
      /(^|\/)node_modules\// ||
      /^panel\/docs\/plans\// ||
      /^tests\/e2e\/results\// ||
      /^tests\/e2e\/panel-driver\/node_modules\// ||
      /^tests\/benchmark\/outputs\// ||
      /^tests\/e2e\/(AUDIT-REPORT|MATRIX|UI-SUITE)\.md$/ ||
      /^nohup\.out$/ ||
      /^trace_err2?\.txt$/ ||
      /^(gsm8k_qwen3_0\.6b_results|vlm_benchmark_results)\.json$/ ||
      /^vmlx_engine\/models\/minimax_m3\/(BUILD-STATUS|MASTER-STATUS|CAMPAIGN-CHECKLIST|CAMPAIGN-PROGRESS-LOG|M3-EAGLE3-NATIVE-MTP-HANDOFF|M3-MOE-QUANT-FIX-HANDOFF)\.md$/ ||
      /^vmlx_engine\/models\/minimax_m3\/MODEL-MATRIX-AUTODETECT\.txt$/ ||
      /^\.agents\// ||
      /^\.agent\// ||
      /^\.claude\// ||
      /^\.codex\// ||
      /^\.sisyphus\// ||
      /^\.factory\// ||
      /(^|\/)(botes|evidence|private-evidence|vmlx-proof|screenshots?|screen-recordings?|cdp-captures?|raw-sse|runtime-logs?)(\/|$)/ ||
      /(^|\/)[^\/]+\.(sqlite|db)$/ {
        print
      }
    ' |
    sort -u
)

if [ -n "$historical_forbidden" ]; then
  printf '%s\n' \
    'ERROR: public repository history contains forbidden private/internal artifacts:' \
    "$historical_forbidden" >&2
  exit 1
fi

sensitive_paths=$(
  git ls-files |
    awk '
      /(^|\/)\.(pypirc|npmrc|netrc)$/ ||
      /(^|\/)\.env($|\.)/ ||
      /\.(p8|p12|pem|key|cer|crt|der|mobileprovision|provisionprofile)$/ ||
      /(^|\/)[^\/]*notary-results?[.]json$/ ||
      /(^|\/)[^\/]*notary-(info|log|submit)[.]json$/ ||
      /(^|\/)[^\/]*(notar|sign|release|credential)[^\/]*[.]local[.][^\/]+$/ ||
      /(^|\/)(notary-results|notary-records|private-release|release-private|release-credentials|signing-secrets)(\/|$)/ {
        if ($0 !~ /[.]example$/) {
          print
        }
      }
    '
)

if [ -n "$sensitive_paths" ]; then
  printf '%s\n' \
    'ERROR: public repository contains tracked release credentials or private signing/notarization material:' \
    "$sensitive_paths" >&2
  exit 1
fi

historical_sensitive_paths=$(
  public_git_log --name-only --format= |
    awk '
      /(^|\/)\.(pypirc|npmrc|netrc)$/ ||
      /(^|\/)\.env($|\.)/ ||
      /\.(p8|p12|pem|key|cer|crt|der|mobileprovision|provisionprofile)$/ ||
      /(^|\/)[^\/]*notary-results?[.]json$/ ||
      /(^|\/)[^\/]*notary-(info|log|submit)[.]json$/ ||
      /(^|\/)[^\/]*(notar|sign|release|credential)[^\/]*[.]local[.][^\/]+$/ ||
      /(^|\/)(notary-results|notary-records|private-release|release-private|release-credentials|signing-secrets)(\/|$)/ {
        if ($0 !~ /[.]example$/) {
          print
        }
      }
    ' |
    sort -u
)

if [ -n "$historical_sensitive_paths" ]; then
  printf '%s\n' \
    'ERROR: public repository history contains release credentials or private signing/notarization material.' >&2
  exit 1
fi

private_host_pattern='erics-m5-'"max"'([0-9]*)?([.]tail[[:alnum:]]+[.]ts[.]net|[.]local)'
private_volume_pattern='/Volumes/'"Erics"'LLMDrive'
private_cache_pattern='[.]cache/vmlx-'"proof"
stale_source_alias_pattern='/(private/)?tmp/vmlx-[0-9][^[:space:]]*-build'

private_strings=$(
  {
    git grep -Il -E \
      "(${private_host_pattern}|${private_volume_pattern}|${private_cache_pattern}|${stale_source_alias_pattern})" \
      -- . ':(exclude)scripts/check-public-repo-hygiene.sh' 2>/dev/null || true
    git grep -InE '/Users/[A-Za-z0-9._-]+' -- . 2>/dev/null |
      grep -Ev '/Users/(example|u)(/|[^A-Za-z0-9._-])' |
      grep -v '^scripts/check-public-repo-hygiene.sh:' |
      awk -F: '{print $1}' || true
  } | sort -u
)

if [ -n "$private_strings" ]; then
  printf '%s\n' \
    'ERROR: public repository contains private host or filesystem paths:' \
    "$private_strings" >&2
  exit 1
fi

invalid_commit_emails=$(
  public_git_log --format='%ae%n%ce' |
    awk '$0 !~ /^[^[:space:]<>@]+@[^[:space:]<>@]+$/ {print}' |
    sort -u
)

if [ -n "$invalid_commit_emails" ]; then
  printf '%s\n' \
    'ERROR: public repository history contains malformed author or committer email metadata.' >&2
  exit 1
fi

private_message_commits=$(
  public_git_log --format='@@%H%n%B' |
    awk \
      -v private_host_pattern="$private_host_pattern" \
      -v private_volume_pattern="$private_volume_pattern" \
      -v private_cache_pattern="$private_cache_pattern" '
      /^@@[0-9a-f]+$/ {
        commit = substr($0, 3)
        next
      }
      {
        line = $0
        while (match(line, /\/Users\/[A-Za-z0-9._-]+/)) {
          candidate = substr(line, RSTART, RLENGTH)
          if (candidate != "/Users/example" && candidate != "/Users/u") {
            bad[commit] = 1
          }
          line = substr(line, RSTART + RLENGTH)
        }
        if (($0 ~ private_host_pattern) || ($0 ~ private_volume_pattern) || ($0 ~ private_cache_pattern)) {
          bad[commit] = 1
        }
      }
      END {
        for (commit in bad) {
          print commit
        }
      }
    ' |
    sort
)

if [ -n "$private_message_commits" ]; then
  printf '%s\n' \
    'ERROR: public history contains private identifiers in commit messages:' \
    "$private_message_commits" >&2
  exit 1
fi

printf '%s\n' 'Public repository hygiene check passed.'
