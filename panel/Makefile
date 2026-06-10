# vMLX panel — composable build targets (run from panel/ or via root delegation)
#
#   make            show this help (default)
# Prefer repo root:  cd .. && make help
#
# Path model: this Makefile owns PANEL; REPO is one level up. Do not invoke
# $(MAKE) -C panel from here — root Makefile delegates panel targets inward.

.DEFAULT_GOAL := help

SHELL := /bin/bash
# Prefer macOS BSD utilities (find, tar, sed) over GNU coreutils from Homebrew/PATH.
export PATH := /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$(PATH)
PANEL := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(PANEL)/..)
BUNDLE_PY := $(PANEL)/bundled-python/python/bin/python3
STAGED_APP := $(PANEL)/release/mac-arm64/vMLX.app
RESOLVE_JANG := $(REPO)/scripts/resolve-jang-tools.sh
JANG_TOOLS := $(or $(VMLX_JANG_TOOLS_SOURCE),$(VMLINUX_JANG_TOOLS_SOURCE),$(shell bash "$(RESOLVE_JANG)" --try "$(REPO)" 2>/dev/null))
JANG_REPO := $(abspath $(or $(JANG_TOOLS),$(HOME)/jang/jang-tools)/..)
JANG_FRESHEN_BRANCH ?= codex/mimo-v25-cache-contract
ELECTRON_BUILDER := CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder
NPM_INSTALL := npm install --ignore-scripts
APP_DEST ?= /Applications/vMLX.app

.PHONY: help doctor deps deps-native check-jang check-bundle \
        bundle sync check-sync verify compile panel app app-rebuild dmg release installable install dev \
        clean clean-bundle clean-deps clean-all uninstall \
        verify-hashes \
        first-dmg engine-app engine-and-install engine-dmg ui-dmg

help:
	@echo "vMLX panel Makefile (see repo root: make help)"
	@echo ""
	@echo "ARTIFACT RULE — one authoritative build at a time (never staged + DMG together)"
	@echo "  Dev:   make install → staged .app; any stale DMG is removed on rebuild"
	@echo "  Ship:  make release → DMG from staged .app; staged dir removed after pack"
	@echo "  Flow:  make install → test in /Applications → make release"
	@echo ""
	@echo "LOCAL DEV (staged .app — no DMG required)"
	@echo "  sync             refresh vmlx_engine + jang in bundled-python/"
	@echo "  app              compile + electron-builder --dir → release/mac-arm64/vMLX.app"
	@echo "  install          sync/app if sources changed + copy/sign to $(APP_DEST)"
	@echo "  engine-app       sync → app  (after vmlx_engine / jang edits)"
	@echo "  engine-and-install  engine-app → install"
	@echo ""
	@echo "DISTRIBUTION (DMG + panel/release/SHA256SUMS)"
	@echo "  release          app → DMG from staged .app → remove staged dir"
	@echo "  installable      alias for release"
	@echo "  engine-dmg       sync → release  (ship after engine/jang edits)"
	@echo "  first-dmg        deps → check-jang → bundle → release"
	@echo ""
	@echo "RESUME AFTER FAILURE (fix the problem, re-run only this)"
	@echo "  bundle failed    → make bundle"
	@echo "  app/pack failed  → make app"
	@echo "  dmg/pack failed  → make release"
	@echo "  engine/jang edit → make engine-app  (local) or make engine-dmg  (ship)"
	@echo "  start over       → make first-dmg"
	@echo ""
	@echo "OTHER"
	@echo "  dev              hot reload, no DMG"
	@echo "  doctor           what's installed / missing"
	@echo "  verify-hashes    check release/SHA256SUMS (optional QA)"
	@echo ""
	@echo "LOW-LEVEL (normally not needed directly)"
	@echo "  deps bundle sync verify compile app clean clean-all uninstall"
	@echo ""
	@echo "NOT npm run build for DMG — that only compiles the panel (see build:full to rebundle)."

doctor:
	@echo "==> vMLX panel build doctor"
	@if [[ -n "$${CONDA_DEFAULT_ENV:-}" ]]; then \
	  echo "  ⚠ conda active: $$CONDA_DEFAULT_ENV — run 'conda deactivate' before panel builds if you hit odd errors"; \
	fi
	@command -v node >/dev/null && echo "  node: $$(node -v)" || echo "  node: MISSING"
	@command -v npm >/dev/null && echo "  npm:  $$(npm -v)" || echo "  npm: MISSING"
	@test -d "$(PANEL)/node_modules" && echo "  node_modules: OK" || echo "  node_modules: missing → make deps"
	@test -x "$(BUNDLE_PY)" && echo "  bundled-python: OK" || echo "  bundled-python: missing → make bundle"
	@if [[ -d "$(JANG_REPO)/.git" ]]; then \
	  branch="$$(git -C "$(JANG_REPO)" branch --show-current 2>/dev/null || echo detached)"; \
	  echo "  jang repo: $(JANG_REPO) ($$branch)"; \
	  if [[ "$$branch" != "$(JANG_FRESHEN_BRANCH)" ]] && [[ ! -f "$(JANG_TOOLS)/jang_tools/step37/__init__.py" ]]; then \
	    echo "  ⚠ jang missing step37 — run: cd .. && make freshen-jang"; \
	  elif [[ ! -f "$(JANG_TOOLS)/jang_tools/step37/__init__.py" ]]; then \
	    echo "  ⚠ jang step37 missing — run: cd .. && make freshen-jang"; \
	  else \
	    echo "  jang step37: OK"; \
	  fi; \
	else \
	  echo "  jang-tools: missing — run: bash $(RESOLVE_JANG) $(REPO)"; \
	  bash "$(RESOLVE_JANG)" "$(REPO)" 2>&1 | sed 's/^/    /' || true; \
	fi
	@test -d "$(PANEL)/dist" && echo "  dist/: present" || echo "  dist/: not built yet → make app or make release"
	@test -d "$(APP_DEST)" && echo "  installed app: $(APP_DEST)" || echo "  installed app: not in /Applications"
	@test -f "$(PANEL)/release/SHA256SUMS" && echo "  SHA256SUMS: present" || echo "  SHA256SUMS: missing → make release"

deps:
	cd $(PANEL) && $(NPM_INSTALL)

deps-native: deps
	cd $(PANEL) && npx electron-builder install-app-deps

check-jang:
	@if [ -z "$(JANG_TOOLS)" ] || [ ! -f "$(JANG_TOOLS)/pyproject.toml" ]; then \
		bash "$(RESOLVE_JANG)" "$(REPO)"; \
	fi
	@test -f "$(JANG_TOOLS)/jang_tools/step37/__init__.py" || { \
		echo "ERROR: $(JANG_TOOLS) is missing jang_tools/step37."; \
		echo "       Run: cd .. && make freshen-jang"; \
		exit 1; \
	}
	@echo "  jang-tools OK: $(JANG_TOOLS)"

check-bundle:
	@test -x "$(BUNDLE_PY)" || { \
		echo "ERROR: bundled-python missing. Run: make bundle"; \
		exit 1; \
	}

bundle: check-jang
	cd $(PANEL) && ./scripts/bundle-python.sh

sync: check-bundle
	cd $(PANEL) && ./scripts/sync-bundled-local-packages.sh

check-sync: check-bundle
	@chmod +x "$(PANEL)/scripts/check-sync-local-packages.sh" \
	  "$(PANEL)/scripts/sources-newer-than.sh" \
	  "$(PANEL)/scripts/sync-bundled-local-packages.sh"
	@"$(PANEL)/scripts/check-sync-local-packages.sh"

verify: check-bundle
	cd $(PANEL) && ./scripts/verify-bundled-python.sh

compile panel:
	cd $(PANEL) && npx electron-vite build

app-rebuild:
	@chmod +x "$(PANEL)/scripts/prune-ship-artifacts.sh"
	@echo "==> Building staged app (missing or sources changed)"
	@PANEL_DIR="$(PANEL)" "$(PANEL)/scripts/prune-ship-artifacts.sh"
	cd $(PANEL) && $(ELECTRON_BUILDER) --mac --dir
	@test -d "$(STAGED_APP)" || { \
	  echo "ERROR: staged app missing after electron-builder --dir"; \
	  echo "       Expected: $(STAGED_APP)"; \
	  exit 1; \
	}
	@echo "✓ staged app ready: $(STAGED_APP)"
	@echo "  Install:  make install"
	@echo "  Ship DMG: make release"

app: deps check-bundle check-sync
	@chmod +x "$(PANEL)/scripts/staged-app-stale.sh" \
	  "$(PANEL)/scripts/sources-newer-than.sh" \
	  "$(PANEL)/scripts/prune-ship-artifacts.sh"
	@if [ -d "$(STAGED_APP)" ] && ! PANEL_DIR="$(PANEL)" REPO_DIR="$(REPO)" "$(PANEL)/scripts/staged-app-stale.sh" "$(STAGED_APP)"; then \
	  if PANEL_DIR="$(PANEL)" STAGED_APP="$(STAGED_APP)" "$(PANEL)/scripts/release-ship-stale.sh"; then \
	    PANEL_DIR="$(PANEL)" "$(PANEL)/scripts/prune-ship-artifacts.sh"; \
	  fi; \
	  echo "==> Staged app up to date: $(STAGED_APP)"; \
	else \
	  $(MAKE) app-rebuild; \
	fi

dmg: app
	@chmod +x "$(PANEL)/scripts/release-ship-stale.sh" \
	  "$(PANEL)/scripts/hash-release-artifacts.sh" \
	  "$(PANEL)/scripts/prune-ship-artifacts.sh"
	@if [ ! -d "$(STAGED_APP)" ]; then \
	  echo "ERROR: staged app missing — run make app first" >&2; \
	  exit 1; \
	fi
	@if PANEL_DIR="$(PANEL)" STAGED_APP="$(STAGED_APP)" "$(PANEL)/scripts/release-ship-stale.sh"; then \
	  echo "==> Building DMG from staged app: $(STAGED_APP)"; \
	  cd $(PANEL) && $(ELECTRON_BUILDER) --mac dmg --prepackaged "$(STAGED_APP)"; \
	  "$(PANEL)/scripts/hash-release-artifacts.sh"; \
	  rm -rf "$(PANEL)/release/mac-arm64"; \
	  echo "==> Removed staged app directory (DMG is the ship artifact)"; \
	else \
	  echo "==> DMG up to date"; \
	fi

release: dmg
installable: release

verify-hashes:
	@test -f "$(PANEL)/release/SHA256SUMS" || { \
		echo "ERROR: $(PANEL)/release/SHA256SUMS missing. Run: make release"; \
		exit 1; \
	}
	@cd "$(PANEL)/release" && shasum -a 256 -c SHA256SUMS

install: deps check-bundle check-sync
	@chmod +x "$(PANEL)/scripts/staged-app-stale.sh" \
	  "$(PANEL)/scripts/release-ship-stale.sh" \
	  "$(PANEL)/scripts/prune-ship-artifacts.sh" \
	  "$(PANEL)/scripts/install-from-release.sh"
	@if [ -d "$(STAGED_APP)" ] && PANEL_DIR="$(PANEL)" STAGED_APP="$(STAGED_APP)" "$(PANEL)/scripts/release-ship-stale.sh"; then \
	  PANEL_DIR="$(PANEL)" "$(PANEL)/scripts/prune-ship-artifacts.sh"; \
	fi
	@need_app=1; \
	if [ -d "$(STAGED_APP)" ] && ! PANEL_DIR="$(PANEL)" "$(PANEL)/scripts/staged-app-stale.sh" "$(STAGED_APP)"; then \
	  need_app=0; \
	  echo "==> Staged app up to date"; \
	elif [ ! -d "$(STAGED_APP)" ] && ! PANEL_DIR="$(PANEL)" STAGED_APP="$(STAGED_APP)" "$(PANEL)/scripts/release-ship-stale.sh"; then \
	  need_app=0; \
	  echo "==> Installing from current DMG (no staged app)"; \
	fi; \
	if [ "$$need_app" = "1" ]; then \
	  $(MAKE) app-rebuild; \
	fi
	@"$(PANEL)/scripts/install-from-release.sh"

dev: deps
	cd $(PANEL) && npm run dev

clean:
	@echo "==> Removing panel build outputs (dist/, release/)"
	rm -rf "$(PANEL)/dist" "$(PANEL)/release"

clean-bundle:
	@echo "==> Removing bundled-python/ (re-run make bundle to recreate)"
	rm -rf "$(PANEL)/bundled-python"

clean-deps:
	@echo "==> Removing node_modules/ (re-run make deps to recreate)"
	rm -rf "$(PANEL)/node_modules"

clean-all: clean clean-bundle clean-deps

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

first-dmg: deps check-jang bundle dmg

engine-app: sync app

engine-and-install: engine-app install

engine-dmg: check-sync release

ui-dmg: release
