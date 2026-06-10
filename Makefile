# vMLX — repo-root build and install entry point
#
#   make            show this help (default)
#   make help       same
#
# Make runs prerequisite targets automatically. Example:
#   make dmg        → deps, check-bundle, then package (verify+compile inside electron-builder)

.DEFAULT_GOAL := help

SHELL := /bin/bash
# Prefer macOS BSD utilities (find, tar, sed) over GNU coreutils from Homebrew/PATH.
export PATH := /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$(PATH)
REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PANEL := $(REPO)/panel
VENV := $(REPO)/.venv
VENV_PY := $(VENV)/bin/python
JANG_FRESHEN_BRANCH ?= codex/mimo-v25-cache-contract
FRESHEN_SCRIPT := $(REPO)/scripts/freshen-repo.sh
RESOLVE_JANG := $(REPO)/scripts/resolve-jang-tools.sh
JANG_TOOLS := $(or $(VMLX_JANG_TOOLS_SOURCE),$(VMLINUX_JANG_TOOLS_SOURCE),$(shell bash "$(RESOLVE_JANG)" --try "$(REPO)" 2>/dev/null))
JANG_REPO := $(abspath $(or $(JANG_TOOLS),$(HOME)/jang/jang-tools)/..)
ENGINE_EXTRAS ?= dev,jang,image
APP_DEST ?= /Applications/vMLX.app

.PHONY: help doctor \
        venv engine-install engine-install-local-jang engine-sync-jang \
        test-engine test-panel test \
        panel panel-help \
        deps bundle sync verify compile app dmg install dev \
        clean clean-venv clean-all uninstall \
        freshen freshen-main freshen-jang freshen-all \
        stash stash-apply stash-pop stash-list \
        verify-hashes \
        first-dmg release installable engine-app engine-and-install engine-dmg ui-dmg

help:
	@echo "vMLX — repository Makefile"
	@echo ""
	@echo "LOCAL DEV (staged .app — no DMG required)"
	@echo "  make sync              refresh vmlx_engine + jang in panel/bundled-python/"
	@echo "  make app               compile + staged panel/release/mac-arm64/vMLX.app"
	@echo "  make install           sync/app if changed + install to $(APP_DEST)"
	@echo "  make engine-app        sync + app  (after vmlx_engine / jang edits)"
	@echo "  make engine-and-install  engine-app + install"
	@echo ""
	@echo "DISTRIBUTION (DMG + panel/release/SHA256SUMS)"
	@echo "  make first-dmg     once: deps + bundle + release"
	@echo "  make release       staged .app → DMG; removes staged dir when done"
	@echo "  make installable   alias for make release"
	@echo "  make engine-dmg    check-sync + release (ship after engine/jang edits)"
	@echo ""
	@echo "RESUME AFTER FAILURE (fix issue, re-run only the failed step)"
	@echo "  bundle step failed     → make bundle"
	@echo "  app / pack step failed → make app"
	@echo "  dmg / pack step failed → make release"
	@echo "  engine/jang changed    → make engine-app  (local) or make engine-dmg  (ship)"
	@echo "  full reset             → make first-dmg"
	@echo ""
	@echo "DESKTOP APP"
	@echo "  doctor           Node, jang, conda warning, bundled-python status"
	@echo "  dev              Electron hot reload (no DMG)"
	@echo "  verify-hashes    optional QA on release/SHA256SUMS"
	@echo ""
	@echo "PYTHON ENGINE (CLI + pytest — uv install is separate, no UI)"
	@echo "  engine-install / engine-install-local-jang / test-engine / test"
	@echo ""
	@echo "UPDATE SOURCE + PR PREP"
	@echo "  freshen-jang / freshen-all     update git checkouts"
	@echo "  STASH_ALL=1 make stash         stash tracked + untracked before freshen"
	@echo "  make stash-apply               restore after freshen"
	@echo ""
	@echo "CLEAN / UNINSTALL"
	@echo "  clean / clean-all / clean-venv / uninstall"
	@echo ""
	@echo "NOTES"
	@echo "  PATH: make prepends /usr/bin, /bin, … before Homebrew (BSD find/tar for scripts)"
	@echo "  jang-tools: ../jang/jang-tools (sibling) or ~/jang/jang-tools — see make doctor"
	@echo "  Local DMG:     make release     (NOT npm run dist — that runs release gates)"
	@echo "  npm run build: panel TS only    (NOT a full Python rebundle)"
	@echo "  npm run build:full              rebundles Python (slow; prefer make bundle)"
	@echo "  Official ship: npm run dist or panel/scripts/build-release-dmgs.sh"
	@echo "  Docs: docs/development/build-test-deploy.md"

doctor:
	@echo "==> vMLX repo doctor"
	@if [[ -n "$${CONDA_DEFAULT_ENV:-}" ]]; then \
	  echo "  ⚠ conda active: $$CONDA_DEFAULT_ENV — consider: conda deactivate"; \
	fi
	@command -v python3 >/dev/null && echo "  python3: $$(python3 --version)" || echo "  python3: MISSING"
	@test -x "$(VENV_PY)" && echo "  .venv: OK ($(VENV_PY))" || echo "  .venv: missing → make engine-install"
	@test -f "$(JANG_TOOLS)/pyproject.toml" && echo "  jang-tools: $(JANG_TOOLS)" || { \
	  echo "  jang-tools: missing"; \
	  bash "$(RESOLVE_JANG)" "$(REPO)" 2>&1 | sed 's/^/    /' || true; \
	}
	@$(MAKE) -C "$(PANEL)" doctor

# ── Python engine (.venv) ─────────────────────────────────────────────

venv:
	@if [ -x "$(VENV_PY)" ]; then \
	  echo "==> .venv already exists: $(VENV_PY)"; \
	else \
	  command -v python3.12 >/dev/null && PY=python3.12 || PY=python3; \
	  echo "==> Creating .venv with $$PY"; \
	  $$PY -m venv "$(VENV)"; \
	fi

engine-install: venv
	@echo "==> Installing vmlx_engine into .venv [.$(ENGINE_EXTRAS)]"
	"$(VENV_PY)" -m pip install -U pip
	"$(VENV_PY)" -m pip install -e "$(REPO)[$(ENGINE_EXTRAS)]"

engine-install-local-jang: engine-install
	@test -f "$(JANG_TOOLS)/pyproject.toml" || { \
		echo "ERROR: jang-tools not found at $(JANG_TOOLS)"; exit 1; \
	}
	@echo "==> Installing local jang-tools into .venv"
	"$(VENV_PY)" -m pip install -e "$(JANG_TOOLS)" --no-deps

engine-sync-jang:
	@test -x "$(VENV_PY)" || { echo "Run: make engine-install"; exit 1; }
	@test -f "$(JANG_TOOLS)/pyproject.toml" || { echo "ERROR: jang-tools not found at $(JANG_TOOLS)"; exit 1; }
	@echo "==> Refreshing local jang-tools in .venv"
	"$(VENV_PY)" -m pip install -e "$(JANG_TOOLS)" --no-deps --force-reinstall

test-engine:
	@test -x "$(VENV_PY)" || { echo "Run: make engine-install"; exit 1; }
	"$(VENV_PY)" -m pytest "$(REPO)/tests/" -k "not Async" -v

test-panel:
	$(MAKE) -C "$(PANEL)" deps
	cd "$(PANEL)" && npx vitest run

test: test-engine test-panel

# ── Update git checkouts ───────────────────────────────────────────────

freshen:
	@test -x "$(FRESHEN_SCRIPT)" || { echo "Missing $(FRESHEN_SCRIPT)"; exit 1; }
	"$(FRESHEN_SCRIPT)" $(if $(filter 1,$(FORCE)),--force,) "$(REPO)" latest-tag

freshen-main:
	@test -x "$(FRESHEN_SCRIPT)" || { echo "Missing $(FRESHEN_SCRIPT)"; exit 1; }
	"$(FRESHEN_SCRIPT)" $(if $(filter 1,$(FORCE)),--force,) "$(REPO)" main

freshen-jang:
	@test -x "$(FRESHEN_SCRIPT)" || { echo "Missing $(FRESHEN_SCRIPT)"; exit 1; }
	@test -d "$(JANG_REPO)/.git" || { \
		echo "ERROR: jang repo not found at $(JANG_REPO)"; \
		echo "       Clone https://github.com/jjang-ai/jangq.git to $(REPO)/../jang"; \
		echo "       Or: export VMLX_JANG_TOOLS_SOURCE=/path/to/jang-tools"; \
		exit 1; \
	}
	"$(FRESHEN_SCRIPT)" $(if $(filter 1,$(FORCE)),--force,) "$(JANG_REPO)" "$(JANG_FRESHEN_BRANCH)"

freshen-all: freshen freshen-jang

# ── Git stash (vmlx working tree) ─────────────────────────────────────

stash:
	@cd "$(REPO)" && { \
	  MSG="$${STASH_MSG:-vmlx make stash $$(date +%Y-%m-%d_%H%M%S)}"; \
	  if [[ "$${STASH_ALL:-0}" == "1" ]]; then \
	    echo "==> Stashing tracked + untracked changes: $$MSG"; \
	    git stash push -u -m "$$MSG"; \
	  else \
	    echo "==> Stashing tracked changes: $$MSG"; \
	    git stash push -m "$$MSG"; \
	  fi; \
	  git stash list | head -3; \
	}

stash-apply:
	@cd "$(REPO)" && { \
	  if [[ -z "$${STASH_INDEX:-}" ]]; then \
	    echo "==> Applying latest stash (keeps entry on stack)"; \
	    git stash apply; \
	  else \
	    echo "==> Applying $${STASH_INDEX}"; \
	    git stash apply "$${STASH_INDEX}"; \
	  fi; \
	}

stash-pop:
	@cd "$(REPO)" && { \
	  if [[ -z "$${STASH_INDEX:-}" ]]; then \
	    echo "==> Popping latest stash"; \
	    git stash pop; \
	  else \
	    echo "==> Popping $${STASH_INDEX}"; \
	    git stash pop "$${STASH_INDEX}"; \
	  fi; \
	}

stash-list:
	@git -C "$(REPO)" stash list

# ── Clean / uninstall ─────────────────────────────────────────────────

clean:
	@$(MAKE) -C "$(PANEL)" clean

clean-venv:
	@echo "==> Removing .venv"
	rm -rf "$(VENV)"

clean-all: clean clean-venv
	@$(MAKE) -C "$(PANEL)" clean-all

uninstall:
	@echo "==> Stopping vMLX if running"
	-pkill -f "vMLX.app" 2>/dev/null || true
	-pkill -f "vmlx-engine" 2>/dev/null || true
	@if [ -d "$(APP_DEST)" ]; then \
		echo "==> Removing $(APP_DEST)"; \
		rm -rf "$(APP_DEST)"; \
	else \
		echo "==> $(APP_DEST) not installed"; \
	fi

# ── Panel / desktop app (delegate to panel/Makefile) ──────────────────

panel-help:
	@$(MAKE) -C "$(PANEL)" help

panel:
	@$(MAKE) -C "$(PANEL)" compile

deps bundle sync check-sync verify compile app dmg release installable install dev \
verify-hashes \
first-dmg engine-app engine-and-install engine-dmg ui-dmg:
	@$(MAKE) -C "$(PANEL)" $@
