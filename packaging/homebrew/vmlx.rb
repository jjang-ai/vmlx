# Homebrew formula for vmlxctl — the vMLX Swift command-line server.
#
# S4 §310 scaffold. Installs the `vmlxctl` binary extracted from the
# notarized Developer ID DMG attached to each vMLX GitHub release,
# shimmed as `vmlx` under `/opt/homebrew/bin` (Apple Silicon) or
# `/usr/local/bin` (Intel). Matches the convention users expect from
# `brew install <tool>` — they get a working `vmlx serve ...` without
# mounting the DMG manually.
#
# Release cadence: the `url` + `sha256` below are updated by the
# release-automation step after the DMG is notarized, mirroring the
# pattern we use for vMLX Electron releases.
#
# Usage (once tapped):
#   brew tap jjang-ai/vmlx
#   brew install vmlx
#   vmlx serve -m /path/to/model
#
# vMLX Swift targets macOS 13+ Apple Silicon. We reject Intel builds
# because MLX's Metal back-end has no Intel macOS codepath.

class Vmlx < Formula
  desc "Local LLM / VLM / image server for Apple Silicon (Swift, MLX)"
  homepage "https://github.com/jjang-ai/vmlx"
  license "MIT"
  version "0.0.0-dev"

  # Filled in by the release script. DMG carries the Developer ID-signed
  # `vmlxctl` binary; we extract just that binary (no bundle) so Homebrew
  # doesn't have to install an app bundle inside its prefix.
  url "https://github.com/jjang-ai/mlxstudio/releases/download/v0.0.0/vmlx-cli-arm64.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

  depends_on :macos
  depends_on arch: :arm64

  def install
    # The archive ships a single executable named `vmlxctl`. Install it
    # under `bin/` and also expose a shorter `vmlx` shim for muscle
    # memory. Homebrew auto-links `bin/*` into the prefix.
    bin.install "vmlxctl"
    bin.install_symlink "vmlxctl" => "vmlx"
  end

  test do
    # A minimal smoke test that proves the binary is executable + prints
    # its version output without trying to load a model (no fixture in
    # the formula's sandbox). `--version` is a trivial arg-only path.
    assert_match(/vmlxctl/i, shell_output("#{bin}/vmlxctl --help"))
  end
end
